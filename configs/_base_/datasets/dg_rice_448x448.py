_base_ = [
    "./rice_source_bs8_448.py",
    "./rice_target_bs8_448.py",
]
train_dataloader = _base_.train_rice_source
val_dataloader = _base_.val_rice_target

test_dataloader = val_dataloader
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator=val_evaluator
