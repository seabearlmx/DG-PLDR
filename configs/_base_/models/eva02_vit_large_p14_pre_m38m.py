# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTEVA02',
        arch='l',
        img_size=224,
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='avg_featmap'),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))