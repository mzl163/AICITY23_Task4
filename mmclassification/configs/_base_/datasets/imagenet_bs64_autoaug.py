_base_ = ['./pipelines/auto_aug.py']

# dataset settings
dataset_type = 'MyDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #dict(type='AutoAugment', policies={{_base_.auto_increasing_policies}}),
    #dict(type='AutoAugment', policies={{_base_.policy_imagenet}}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,#调整batchsize
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='/home/ldl/AiCity/data_process/data_classfication/train',#***************
        ann_file='/home/ldl/AiCity/data_process/data_classfication/train.txt',#****************
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/ldl/AiCity/data_process/data_classfication/val',#******************
        ann_file='/home/ldl/AiCity/data_process/data_classfication/val.txt',#***************
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/home/ldl/AiCity/data_process/data_classfication/val',#********************
        ann_file='/home/ldl/AiCity/data_process/data_classfication/val.txt',#*******************
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
