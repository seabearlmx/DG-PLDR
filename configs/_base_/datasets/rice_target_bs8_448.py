# dataset settings
rice_trg_dataset_type = 'Rice'
rice_trg_data_preprocessor = dict(
    num_classes=3,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

rice_trg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

rice_trg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

rice_trg_train_rice_target = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=rice_trg_dataset_type,
        data_root='/extra_disk/Benchmarks/Rice_Trg',
        split='train',
        pipeline=rice_trg_train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_rice_target = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=rice_trg_dataset_type,
        data_root='/extra_disk/Benchmarks/Rice_Trg',
        split='val',
        pipeline=rice_trg_test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator_rice_target = dict(type='Accuracy', topk=(1, ))

test_rice_target = val_rice_target
test_evaluator_rice_target = val_evaluator_rice_target
