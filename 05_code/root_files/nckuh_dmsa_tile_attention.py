"""Compare uniform and learned pooling of cached kidney image tiles."""
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_matched_tile_control as cache
from nckuh_dmsa_cluster_ci import cluster_intervals

BASE = cache.BASE
OUT = BASE.ROOT / "02_experiments/NCKUH_DMSA_tile_attention_20261103"


class TileHead(nn.Module):
    def __init__(self, learned):
        super().__init__()
        self.learned = learned
        self.projection = nn.Linear(32, 32)
        self.dropout = nn.Dropout(.15)
        self.gate = nn.Linear(32, 1, bias=False)
        self.classifier = nn.Linear(32, 1)

    def forward(self, values, valid):
        h = self.dropout(torch.relu(self.projection(values)))
        raw = self.gate(torch.tanh(h)).squeeze(-1) if self.learned else torch.zeros_like(valid, dtype=h.dtype)
        attention = torch.softmax(raw.masked_fill(~valid, -torch.inf), dim=1)
        pooled = (attention[..., None] * h).sum(1)
        return self.classifier(pooled).squeeze(-1), attention


def make_bags(tiles, offsets, train):
    lengths = np.diff(offsets)
    assert (lengths > 0).all() and offsets[-1] == len(tiles)
    image_index = np.repeat(np.arange(len(lengths)), lengths)
    pca = PCA(32, whiten=True, svd_solver="full").fit(tiles[np.isin(image_index, train)])
    projected = pca.transform(tiles).astype(np.float32)
    bags = np.zeros((len(lengths), int(lengths.max()), 32), np.float32)
    mask = np.zeros(bags.shape[:2], bool)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        bags[i, :end-start] = projected[start:end]
        mask[i, :end-start] = True
    return torch.from_numpy(bags).cuda(), torch.from_numpy(mask).cuda()


def train_head(bags, valid, labels, weights, seed, learned):
    torch.manual_seed(seed)
    model = TileHead(learned).to(bags.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        score, _ = model(bags, valid)
        loss = (nn.functional.binary_cross_entropy_with_logits(score, labels, reduction="none") * weights).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
    return model.eval()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, kidneys, images = cache.tile.load_reference()
    with np.load(cache.OUT / "matched_embeddings.npz") as bundle:
        np.testing.assert_array_equal(bundle["image_ids"], images.image_id)
        tiles, offsets = bundle["tiles"], bundle["offsets"]
    assert tiles.shape == (3277, 1024) and len(offsets) == 821
    roles = pd.read_csv(BASE.OUT / "patient_assignments.csv")
    old = pd.read_csv(cache.OUT / "development_predictions.csv")
    old = old[old.group.isin(["aligned_fusion", "tile_mean", "tile_mean_fusion"])]
    records, diagnostics = [old], []
    y = images.kidney_APN.to_numpy(int)
    for seed in BASE.SEEDS:
        mapping = roles[roles.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(images.patient_alias.map(mapping).eq("train"))
        dev = np.flatnonzero(images.patient_alias.map(mapping).eq("development"))
        target = kidneys[kidneys.patient_alias.map(mapping).eq("development")]
        assert target.patient_alias.nunique() == 42 and images.iloc[train].patient_alias.nunique() == 126
        bags, valid = make_bags(tiles, offsets, train)
        labels = torch.as_tensor(y[train], dtype=torch.float32, device="cuda")
        weights = torch.as_tensor(BASE.image_weights(images, train, y), dtype=torch.float32, device="cuda")
        ref = old[old.seed.eq(seed) & old.group.eq("aligned_fusion")].set_index("kidney_alias").loc[target.kidney_alias]
        np.testing.assert_array_equal(ref.patient_alias, target.patient_alias)
        np.testing.assert_array_equal(ref.label, target.kidney_APN)
        for name, learned in (("mlp_uniform", False), ("mlp_attention", True)):
            model = train_head(bags[train], valid[train], labels, weights, seed, learned)
            with torch.inference_mode():
                logits, attention = model(bags[dev], valid[dev])
                score = BASE.aggregate_image_logits(logits.sigmoid().cpu().numpy(), images.iloc[dev],
                                                    target.kidney_alias)
                entropy = -(attention * attention.clamp_min(1e-8).log()).sum(1)
                n = valid[dev].sum(1)
                diagnostics.append(dict(seed=seed, group=name,
                    normalized_entropy=float((entropy[n > 1] / n[n > 1].float().log()).mean())))
            for group, s in ((name, score), (name + "_fusion", .75 * ref.score.to_numpy() + .25 * score)):
                records.append(target[["patient_alias", "kidney_alias", "side", "DMSA_APN"]].assign(
                    seed=seed, group=group, label=target.kidney_APN.astype(int), score=s))
    pred = pd.concat(records, ignore_index=True)
    ci, delta, _ = cluster_intervals(pred, list(pred.group.unique()), BASE.SEEDS, "tile_mean_fusion")
    _, pooling, _ = cluster_intervals(pred, ["mlp_uniform", "mlp_attention"], BASE.SEEDS, "mlp_uniform")
    OUT.mkdir()
    pred.to_csv(OUT / "development_predictions.csv", index=False)
    ci.to_csv(OUT / "development_ci.csv", index=False)
    pd.concat([delta, pooling]).to_csv(OUT / "paired_delta_ci.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "attention_diagnostics.csv", index=False)
    print(ci.pivot(index="group", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
