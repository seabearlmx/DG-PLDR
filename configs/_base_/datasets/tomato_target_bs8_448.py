# dataset settings
tomato_trg_dataset_type = 'Tomato'
tomato_trg_data_preprocessor = dict(
    num_classes=9,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

tomato_trg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

tomato_trg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_tomato_target = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=tomato_trg_dataset_type,
        data_root='/extra_disk/Benchmarks/Tomato_Trg',
        split='train',
        pipeline=tomato_trg_train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_tomato_target = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=tomato_trg_dataset_type,
        data_root='/extra_disk/Benchmarks/Tomato_Trg',
        split='val',
        pipeline=tomato_trg_test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator_tomato_target = dict(type='Accuracy', topk=(1, ))

test_tomato_target = val_tomato_target
test_evaluator_tomato_target = val_evaluator_tomato_target
