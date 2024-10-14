# dataset settings
plant_src_dataset_type = 'Plant'
plant_src_data_preprocessor = dict(
    num_classes=27,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

plant_src_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

plant_src_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224),
    # dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_plant_source = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=plant_src_dataset_type,
        data_root='/extra_disk/Benchmarks/Plant_Src',
        split='train',
        pipeline=plant_src_train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_plant_source = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=plant_src_dataset_type,
        data_root='/extra_disk/Benchmarks/Plant_Src',
        split='val',
        pipeline=plant_src_test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator_plant_source = dict(type='Accuracy', topk=(1, ))

test_plant_source = val_plant_source
test_evaluator_plant_source = val_evaluator_plant_source
