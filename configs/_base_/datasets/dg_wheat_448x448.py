_base_ = [
    "./wheat_source_bs8_448.py",
    "./wheat_target_bs8_448.py",
]
train_dataloader = _base_.train_wheat_source
val_dataloader = _base_.val_wheat_target

test_dataloader = val_dataloader
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator=val_evaluator
