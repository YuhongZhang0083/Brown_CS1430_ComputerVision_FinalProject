
'''
This file will contain the code for training and testing the MCNN model on your dataset.
'''
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataloader import CrowdDataset
from model import MCNN
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter


# Create a SummaryWriter
# writer = SummaryWriter('runs/experiment_1')

# Set hyperparameters and paths
batch_size = 32
num_epochs = 50
learning_rate = 0.00001
momentum = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.version.cuda)

# Check if GPU is available
print("Is GPU available?", torch.cuda.is_available())
# Get the number of available GPUs
print("Number of GPUs:", torch.cuda.device_count())
# Get the name of the current GPU
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

train_data_dir = '/users/yzhan709/DKLiang_CrowdCounting/MCNN_practice_code/data/original/shanghaitech/part_B_final/train_data/images'
train_gt_dir = '/users/yzhan709/DKLiang_CrowdCounting/MCNN_practice_code/data/original/shanghaitech/part_B_final/train_data/density_B_train'
model_save_path = '/users/yzhan709/DKLiang_CrowdCounting/MCNN_practice_code/savedModel'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Data loading and processing
train_dataset = CrowdDataset(train_data_dir, train_gt_dir, transform=ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialize model, loss function, and optimizer
model = MCNN().to(device)
criterion = torch.nn.MSELoss()

# use Adam optimizer or SGD optimizer
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Train the model
print(f"Number of batches: {len(train_dataloader)}")
print("Starting training...")
# Train the model
for epoch in range(num_epochs):
    model.train()
    # epoch_loss = 0
    epoch_mse_loss = 0
    epoch_mae_loss = 0

    for i, (images, density_maps) in enumerate(train_dataloader):
        print(f"Processing batch {i+1}")
        images = images.to(device)
        density_maps = density_maps.to(device)
        density_maps = density_maps.mean(dim=1, keepdim=True)#Convert the target tensor from RGB to grayscale
        density_maps = F.interpolate(density_maps, size=(120, 160), mode='bilinear', align_corners=False)

        # Forward pass
        predicted_density_maps = model(images)

        # Compute MSE loss
        mse_loss = criterion(predicted_density_maps, density_maps)
        epoch_mse_loss += mse_loss.item()

        # Compute MAE loss
        mae_loss = torch.mean(torch.abs(predicted_density_maps - density_maps))
        epoch_mae_loss += mae_loss.item()

        # # Inside the training loop, after computing the loss
        # writer.add_scalar('training mse_loss', mse_loss.item(), global_step)
        # writer.add_scalar('training mae_loss', mae_loss.item(), global_step)

        # # Update the global step
        # global_step += 1

        # Backward pass and optimization
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_dataloader)}], MSE Loss: {mse_loss.item()}, MAE Loss: {mae_loss.item()}')

    # Save model after each epoch
    torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch+1}.pth'))
    print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {epoch_mse_loss / len(train_dataset)}, MAE Loss: {epoch_mae_loss / len(train_dataset)}')

# writer.close()