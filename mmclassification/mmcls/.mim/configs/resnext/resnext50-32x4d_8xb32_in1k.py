_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet_bs64_autoaug.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
