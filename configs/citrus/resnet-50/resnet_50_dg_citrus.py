_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/dg_citrus_448x448.py',
    '../_base_/schedules/resnet_50.py',
    '../_base_/default_runtime.py'
]

#####################################
# model setting

num_classes = 2

model = dict(
    backbone=dict(
        frozen_stages=-1,   # not freeze backbone  -1   # freeze backbone 4
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/lmx/PD_VFMs/PDDG_v1/mmpretrain/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(
        num_classes=num_classes,
    )
)


#####################################
# dataset setting

input_size = 224
src_data_root = '/extra_disk/Benchmarks/Citrus_Src'
trg_data_root = '/extra_disk/Benchmarks/Citrus_Trg'
metainfo = {
    'classes': ['Greening', 'Healthy'],
}

data_preprocessor = dict(
    num_classes=num_classes,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=input_size),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1)
        ],
        prob=0.8
    ),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=input_size),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type='Citrus',
        data_root=src_data_root,
        metainfo=metainfo,
        # data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='Citrus',
        data_root=trg_data_root,
        metainfo=metainfo,
        # data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='Accuracy'),
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'])
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator

#####################################
# optimizer

max_epoch = 200
warm_up_epoch = 5

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
    # optimizer=dict(type='Adam', lr=0.001)
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=warm_up_epoch,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch - warm_up_epoch,
        by_epoch=True,
        begin=warm_up_epoch,
        end=max_epoch,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epoch, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)

#####################################
# logger

# configure default hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=1, save_best='auto'),
)

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)