"""A pre-registered public-COCO initialization control on the same patient split."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from finetune_patient_heldout_kidney_segmentation import OUT, SEED, digest, dump, verify, read_polygon, rasterize, prediction_mask, overlap_metrics, summarize

PRETRAINED = OUT / 'public_initialization/yolo11n-seg.pt'


def train():
    import torch
    from ultralytics import YOLO
    verify()
    if (OUT / 'coco_control_protocol.json').exists():
        raise FileExistsError('Public-control protocol already locked')
    torch.set_num_threads(4)
    PRETRAINED.parent.mkdir(exist_ok=True)
    model = YOLO(str(PRETRAINED))
    params = dict(epochs=20, imgsz=512, batch=16, freeze=0, optimizer='AdamW', lr0=.001, lrf=.1,
                  patience=6, seed=SEED, degrees=7, translate=.05, scale=.1, fliplr=.5, flipud=0,
                  mosaic=0, mixup=0, hsv_h=0, hsv_s=0, hsv_v=.1)
    protocol = dict(purpose='Resolve undocumented original initialization using a public COCO-pretrained segmentation control. Not random initialization and not a new named model.',
                    checkpoint=str(PRETRAINED), sha256=digest(PRETRAINED), architecture='YOLO11n-seg',
                    official_source='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt',
                    documented_training_source='COCO', documentation='https://docs.ultralytics.com/models/yolo11/',
                    split_manifest_sha256=digest(OUT / 'manifest_deidentified.csv'), parameters=params,
                    selection='Validation box mAP50-95 + mask mAP50-95; no test selection.',
                    evaluation='Same held-out patients and fixed grayscale inference, imgsz640/conf.25/largest polygon/no fallback. Patient mean IoU, missing predictions=0.',
                    qualification='Different architecture and optimization from the reused-model experiment. Not a controlled architecture-only comparison or external clinical validation.')
    dump(OUT / 'coco_control_protocol.json', protocol)
    dump(OUT / 'coco_control_lock.json', dict(protocol_sha256=digest(OUT / 'coco_control_protocol.json')))
    model.train(data=str(OUT / 'data.yaml'), project=str(OUT), name='coco_control', exist_ok=False,
                **params, device=0, workers=0, cache=True, amp=True, plots=False, verbose=False,
                deterministic=True, warmup_epochs=1, weight_decay=.0001, close_mosaic=0, save=True)
    dump(OUT / 'coco_control_training_complete.json', dict(best_checkpoint=str(OUT/'coco_control/weights/best.pt'),
         sha256=digest(OUT/'coco_control/weights/best.pt')))


def evaluate():
    from ultralytics import YOLO
    verify()
    assert digest(OUT / 'coco_control_protocol.json') == json.loads((OUT / 'coco_control_lock.json').read_text())['protocol_sha256']
    if (OUT / 'coco_control_test_results.json').exists():
        raise FileExistsError('Control test already evaluated')
    checkpoint = OUT / 'coco_control/weights/best.pt'
    model = YOLO(str(checkpoint))
    df = pd.read_csv(OUT / 'manifest_deidentified.csv')
    test = df[df.role == 'test']
    rows = []
    masks = OUT / 'test_predictions/coco_control'
    masks.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(test.itertuples()):
        img = cv2.imread(row.image_path)
        gt = rasterize(read_polygon(row.label_path), img.shape)
        pred = model.predict(img, imgsz=640, conf=.25, iou=.7, device=0, verbose=False)[0]
        polygons = [] if pred.masks is None else pred.masks.xy
        mask = prediction_mask(polygons, gt.shape)
        iou, dice = overlap_metrics(mask, gt)
        cv2.imwrite(str(masks / f'{row.image_id}.png'), mask*255)
        rows.append(dict(site=row.site, patient_alias=row.patient_alias, image_id=row.image_id, iou=iou, dice=dice, instances=len(polygons)))
        if i % 40 == 0:
            print(f'Public COCO control test: {i+1}/{len(test)}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'coco_control_test_per_image.csv', index=False)
    patients = result.groupby(['site','patient_alias']).iou.mean()
    rng = np.random.default_rng(SEED)
    groups = [g.to_numpy() for _, g in patients.groupby(level='site')]
    bs = [np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups]).mean() for _ in range(10000)]
    report = dict(checkpoint_sha256=digest(checkpoint), initialization_sha256=digest(PRETRAINED),
                  overall=summarize(result), by_site={s:summarize(g) for s,g in result.groupby('site')},
                  patient_mean_iou_ci95=np.quantile(bs,[.025,.975]).tolist(),
                  target_passed=bool(patients.mean() > .75), documented_initialization='Public COCO-pretrained YOLO11n-seg',
                  qualification='Same fixed patient split held out from kidney-contour training and checkpoint selection. Existing supplied contour provenance and available-frame cropping limitations still apply. Not external clinical validation.')
    dump(OUT / 'coco_control_test_results.json', report)
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','evaluate'])
    args=parser.parse_args()
    {'train':train,'evaluate':evaluate}[args.stage]()
