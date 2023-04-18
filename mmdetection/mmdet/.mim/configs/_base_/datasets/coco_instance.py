# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/ldl/AiCity/make_data/coco_instance/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,poly2mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/home/ldl/AiCity/make_data/coco_instance/train.json',
        img_prefix='/home/ldl/AiCity/make_data/coco_instance/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/home/ldl/AiCity/make_data/coco_instance/val.json',
        img_prefix='/home/ldl/AiCity/make_data/coco_instance/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/ldl/AiCity/make_data/coco_instance/val.json',
        img_prefix='/home/ldl/AiCity/make_data/coco_instance/val/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
