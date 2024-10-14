_base_ = [
    '../../_base_/models/eva02_vit_large_p14_pre_m38m.py',
    '../../_base_/datasets/apple_source_bs8_448.py',
    '../../_base_/schedules/eva02.py',
    '../../_base_/default_runtime.py'
]

#####################################
# model setting

num_classes = 8

model = dict(
    backbone=dict(
        frozen_stages=23,   # not freeze backbone  -1   # freeze backbone 23
        init_cfg=[dict(
            type='Pretrained',
            checkpoint='/extra_disk/PDDG/pretrain_checkpoint/checkpoints/eva02-large-p14_pre_m38m_20230505-b8a1a261.pth',
            prefix='backbone',
        )]),
    head=dict(
        num_classes=num_classes,
    )
)


#####################################
# dataset setting

input_size = 224
src_data_root = '/extra_disk/Benchmarks/Apple_Src'
# trg_data_root = '/extra_disk/Benchmarks/Apple_Trg'
metainfo = {
    'classes': ['Alternaria leaf spot', 'Brown spot', 'Frogeye_spot', 'Grey spot', 'Healthy', 'Mosaic',
                      'Rust', 'Scab'],
}

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # convert image from BGR to RGB
    to_rgb=True,
    num_classes=num_classes,
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
        type='Apple',
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
        type='Apple',
        data_root=src_data_root,
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
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
#     # optimizer=dict(type='Adam', lr=0.001)
# )

# learning policy
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=1e-4,
#         by_epoch=True,
#         begin=0,
#         end=warm_up_epoch,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=max_epoch - warm_up_epoch,
#         by_epoch=True,
#         begin=warm_up_epoch,
#         end=max_epoch,
#         eta_min=1e-6,
#         convert_to_iter_based=True)
# ]

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

find_unused_parameters = True
