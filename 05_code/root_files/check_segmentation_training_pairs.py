"""Label-blind visual and file-integrity audit of the locked segmentation split."""
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from finetune_patient_heldout_kidney_segmentation import OUT, verify, read_polygon, rasterize


verify()
manifest = pd.read_csv(OUT / 'manifest_deidentified.csv')
checked = []
for site in ['CYCH', 'NCKUH']:
    rows = manifest[(manifest.site == site) & (manifest.role == 'train')]
    selected = rows.iloc[np.linspace(0, len(rows)-1, 4).astype(int)]
    checked.extend(selected.to_dict('records'))
canvas = Image.new('RGB', (1200, 1056), 'white')
draw = ImageDraw.Draw(canvas)
for i, row in enumerate(checked):
    img = cv2.imread(row['image_path'])
    gt = rasterize(read_polygon(row['label_path']), img.shape)
    contours = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (20, 220, 80), 1)
    for k, array in enumerate([img, overlay]):
        panel = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        panel.thumbnail((540, 214))
        x, y = (i % 2)*600+k*290, (i // 2)*264+34
        # Separate original and overlay; preserve the available frame's aspect ratio.
        panel.thumbnail((284, 214))
        canvas.paste(panel, (x+(284-panel.width)//2, y+(214-panel.height)//2))
    draw.text(((i % 2)*600+10, (i // 2)*264+9), f"{row['site']} {row['image_id']}: clean / supplied outline", fill='black')
path = OUT / 'training_pair_visual_qa.png'
canvas.save(path)
audit = dict(patients=int(manifest.patient_alias.nunique()), images=len(manifest),
             patients_spanning_roles=int(manifest.groupby('patient_alias').role.nunique().gt(1).sum()),
             exact_decoded_images_spanning_roles=int(manifest.groupby('decoded_grayscale_sha256').role.nunique().gt(1).sum()),
             supplied_label_files_unchanged=all(hashlib.sha256(open(p,'rb').read()).hexdigest()==h for p,h in zip(manifest.label_path,manifest.label_sha256)),
             visual_qa_scope='Eight systematic training-only examples, four per site. No held-out images used to change data or model.',
             visual_qa_file=str(path))
(OUT / 'data_integrity_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
