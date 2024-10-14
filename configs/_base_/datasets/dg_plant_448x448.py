_base_ = [
    "./plant_source_bs8_448.py",
    "./plant_target_bs8_448.py",
]
train_dataloader = _base_.train_plant_source
val_dataloader = _base_.val_plant_target

test_dataloader = val_dataloader
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator=val_evaluator
