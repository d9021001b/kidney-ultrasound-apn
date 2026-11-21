"""Package verified local checkpoints, evidence, figures and the technical deck."""
from pathlib import Path
import json
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import torch
import ultralytics
from finetune_patient_heldout_kidney_segmentation import ROOT, OUT, PRETRAINED, digest


def metadata(path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    model = c.get('model')
    return dict(sha256=digest(path), date=c.get('date'), ultralytics_version=c.get('version'),
                architecture=model.yaml.get('yaml_file'), classes=model.yaml.get('nc'),
                class_names=model.names, recorded_training_data=c.get('train_args',{}).get('data'))


def main():
    destination = ROOT / '01_submission/CYCH_segmentation_finetune_20261004'
    models = destination / 'Models'
    models.mkdir(exist_ok=True)
    deck = ROOT / '01_submission/CYCH_technical_presentation_20261004/CYCH_technical_explanation_zhTW_v3_11configs_segmentation_20261004.pptx'
    assert deck.exists(), 'Finalized deck must exist before packaging'
    shutil.copy2(deck, destination / deck.name)
    files = {'reused_yolov12x_seg_finetuned': ('finetune', 'YOLOv12x_seg_CYCH_NCKUH_finetuned_best.pt'),
             'public_coco_yolo11n_seg_finetuned': ('coco_control', 'YOLO11n_seg_COCO_CYCH_NCKUH_finetuned_best.pt')}
    registry = {'executed_versions': {'torch':torch.__version__, 'ultralytics':ultralytics.__version__},
                'original_reused_initialization':metadata(PRETRAINED),
                'public_coco_initialization':metadata(OUT/'public_initialization/yolo11n-seg.pt'),
                'fine_tuned':{}, 'scope':'Kidney contour prediction on existing kidney-centred available source frames. Not inflammation detection or external clinical validation.'}
    for key, (folder, filename) in files.items():
        source = OUT / folder / 'weights/best.pt'
        target = models / filename
        shutil.copy2(source, target)
        assert digest(source) == digest(target)
        table = pd.read_csv(OUT / folder / 'results.csv')
        fitness = table['metrics/mAP50-95(B)'] + table['metrics/mAP50-95(M)']
        registry['fine_tuned'][key] = dict(file=filename, metadata=metadata(target),
            actual_epochs=int(table.epoch.max()), selected_epoch=int(table.loc[fitness.idxmax(),'epoch']),
            validation_fitness_definition='box mAP50-95 + mask mAP50-95')
        shutil.copy2(OUT/folder/'args.yaml', destination / f'{folder}_executed_args.yaml')
        shutil.copy2(OUT/folder/'results.csv', destination / f'{folder}_development_training.csv')
    (destination / 'model_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    shutil.copy2(destination/'model_registry.json',OUT/'model_registry.json')
    for name in ['test_results.json','test_per_image_deidentified.csv','test_per_patient_deidentified.csv',
                 'protocol.json','protocol_lock.json','data_integrity_audit.json',
                 'coco_control_protocol.json','coco_control_lock.json','coco_control_test_results.json',
                 'coco_control_test_per_image.csv','README.txt']:
        shutil.copy2(OUT/name, destination/name)
    source_folder=destination/'Source'
    source_folder.mkdir(exist_ok=True)
    for f in (OUT/'source').glob('*.py'):
        shutil.copy2(f,source_folder/f.name)
    bundle=ROOT/'01_submission/CYCH_segmentation_and_technical_v3_20261004.zip'
    if bundle.exists():
        raise FileExistsError(bundle)
    with ZipFile(bundle,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(destination.rglob('*')):
            if p.is_file():
                archive.write(p,p.relative_to(destination.parent))
    with ZipFile(bundle) as archive:
        assert archive.testzip() is None
    print(json.dumps(dict(bundle=str(bundle),bytes=bundle.stat().st_size,sha256=digest(bundle),registry=registry),indent=2))


if __name__=='__main__':
    main()
