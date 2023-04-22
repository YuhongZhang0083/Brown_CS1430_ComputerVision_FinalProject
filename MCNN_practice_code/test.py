import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataloader import CrowdDataset
from model import MCNN

# Set hyperparameters and paths
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_dir = 'path/to/test/data'
test_gt_dir = 'path/to/test/ground_truth'
model_load_path = 'path/to/save/model'
results_save_path = 'path/to/save/results'

if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)

# Data loading and processing
test_dataset = CrowdDataset(test_data_dir, test_gt_dir, transform=ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model and load weights
model = MCNN().to(device)
model.load_state_dict(torch.load(model_load_path))

model.eval()
total_mae = 0
total_mse = 0

with torch.no_grad():
    for i, (images, density_maps) in enumerate(test_dataloader):
        images = images.to(device)
        density_maps = density_maps.to(device)

        # Forward pass
        predicted_density_maps = model(images)

        # Calculate performance metrics
        mae = torch.abs(predicted_density_maps - density_maps).mean()
        mse = torch.pow(predicted_density_maps - density_maps, 2).mean()

        total_mae += mae.item()
        total_mse += mse.item()

print(f'MAE: {total_mae / len(test_dataset)}, MSE: {total_mse / len(test_dataset)}')
