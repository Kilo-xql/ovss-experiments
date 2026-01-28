
custom_imports = dict(imports=['proxyclip_segmentor'], allow_failed_imports=False)
# cfg_loveda_lmdb_0_200.py

_base_ = r'/mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/repo/ProxyCLIP/configs/base_config.py'
# 让 eval.py 最后能打印/记录，不然会 KeyError
dataset_type = 'LoveDA'
dataset_name = 'loveda_lmdb_0_200'

# ====== dataset ======
metainfo = dict(
    classes=('background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture')
)

data_root = r'/mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/datasets_ref/loveda_lmdb_0_200'

# 关键：必须有 PackSegInputs，模型才会收到 data['inputs']
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

val_dataloader = dict(
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file=rf'{data_root}/split.txt',
        metainfo=metainfo,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# 这两个必须存在（和 val_dataloader/val_evaluator 成套）
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ====== model ======
model = dict(
    type='ProxyCLIPSegmentation',
    clip_type='openai',
    model_type='ViT-B/16',
    vfm_model='sam',
    name_path=r'/mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/repo/ProxyCLIP/configs/cls_loveda.txt',
    checkpoint=r'/mnt/e/ovss_project/pythonProject1/ovss_experiments_repo/scripts/proxyclip/checkpoints/sam/sam_vit_b_01ec64.pth',
)
