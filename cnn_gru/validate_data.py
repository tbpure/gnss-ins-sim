import torch

data = torch.load("val_info/20250603_0946/train_data.pth")
input = data["input_train"]
target = data["output_train"]
for i in input.shape[0]:
    this_input = input[i].permute(2, 0, 1)
    this_target = target[i].squeeze()
    real = this_target.detach().cpu().numpy()
    real_trajectory = real.cumsum(axis=0)
    pass
pass