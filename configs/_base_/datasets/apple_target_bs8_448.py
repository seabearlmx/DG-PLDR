# dataset settings
apple_target_dataset_type = 'Apple'
apple_target_data_preprocessor = dict(
    num_classes=8,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

apple_target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

apple_target_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_apple_target = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=apple_target_dataset_type,
        data_root='/extra_disk/Benchmarks/Apple_Trg',
        split='train',
        pipeline=apple_target_train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_apple_target = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=apple_target_dataset_type,
        data_root='/extra_disk/Benchmarks/Apple_Trg',
        split='val',
        pipeline=apple_target_test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator_apple_target = dict(type='Accuracy', topk=(1, ))

test_apple_target = val_apple_target
test_evaluator_apple_target = val_evaluator_apple_target
