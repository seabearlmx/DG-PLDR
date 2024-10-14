# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=160,
        by_epoch=True,
        begin=40,
        end=200,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type='EpochBasedTrainLoop', by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()

#default_hooks = dict(
    # only keeps the latest 3 checkpoints
#    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

#randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
#resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)
