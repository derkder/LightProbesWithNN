import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import OpenEXR
import Imath
import json
import os

# Function to save EXR file
def save_exr(file_path, rgb_data):
    height, width, _ = rgb_data.shape
    header = OpenEXR.Header(width, height)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # Split the RGB data into separate channels
    r = rgb_data[:, :, 0].astype(np.float32).tobytes()
    g = rgb_data[:, :, 1].astype(np.float32).tobytes()
    b = rgb_data[:, :, 2].astype(np.float32).tobytes()
    
    # Create an EXR file and write the channels
    exr_file = OpenEXR.OutputFile(file_path, header)
    exr_file.writePixels({'R': r, 'G': g, 'B': b})
    exr_file.close()

# Function to read EXR file
def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    # Read the RGB channels
    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, (size[1], size[0])) for c in rgb]  # Note the order of size
    
    # Stack the channels to form an (H, W, 3) array
    stacked_rgb = np.stack(rgb, axis=-1)
    
    # Crop to the first 500 rows and 500 columns
    cropped_rgb = stacked_rgb[:500, :500, :]
    
    return cropped_rgb

# Function to generate primaryRayOrigin and rayDir
def generate_rays(frame_dim, radius, kProbeLoc):
    hit_points = []
    ray_dirs = []
    for y in range(frame_dim[1]):
        for x in range(frame_dim[0]):
            theta = np.pi * y / frame_dim[1]
            phi = 2 * np.pi * x / frame_dim[0]
            x_val = radius * np.sin(theta) * np.cos(phi)
            z_val = radius * np.sin(theta) * np.sin(phi)
            y_val = radius * np.cos(theta)
            hit_point = np.array([x_val, y_val, z_val]) + kProbeLoc
            ray_dir = np.array([x_val, y_val, z_val])
            ray_dir = - ray_dir / np.linalg.norm(ray_dir)
            hit_points.append(hit_point)
            ray_dirs.append(ray_dir)
    return np.array(hit_points), np.array(ray_dirs)

# Custom Dataset
class LightProbeDataset(Dataset):
    def __init__(self, exr_files, json_file, frame_dim, radius):
        self.exr_files = exr_files
        self.frame_dim = frame_dim
        self.radius = radius
        with open(json_file, 'r') as f:
            self.kProbeLocs = json.load(f)
        for i in range(len(self.kProbeLocs)):
            self.kProbeLocs[i] = np.array([self.kProbeLocs[i]['new_x'], self.kProbeLocs[i]['new_y'], self.kProbeLocs[i]['new_z']])
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for i, exr_file in enumerate(self.exr_files):
            kProbeLoc = np.array(self.kProbeLocs[i])
            rgb_values = read_exr(exr_file)
            hit_points, ray_dirs = generate_rays(self.frame_dim, self.radius, kProbeLoc)
            for j in range(self.frame_dim[0] * self.frame_dim[1]):
                data.append((hit_points[j], ray_dirs[j], rgb_values[j // self.frame_dim[1], j % self.frame_dim[0]], i))
            print(f"Processed {i+1}/{len(self.exr_files)}")
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hit_point, ray_dir, rgb, group_idx = self.data[idx]
        return torch.tensor(hit_point, dtype=torch.float32), torch.tensor(ray_dir, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32), group_idx

class LightProbeNN(nn.Module):
    def __init__(self):
        super(LightProbeNN, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, hit_point, ray_dir):
        x = torch.cat((hit_point, ray_dir), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load Data
test_sample_count = 3  # Number of test samples
test_exr_files = [f"D:/Projects/LightProbesWithNN/dumped_data/dim500/test/frame_{i:04d}/Mogwai.AccumulatePass.output.100000.exr" for i in range(test_sample_count)]
test_json_file = "D:/Projects/LightProbesWithNN/dumped_data/dim500/test/info.json"
model_file = "D:/Projects/LightProbesWithNN/NNAttemps/BasicNN/models/best_light_probe_model3.pth"
frame_dim = (500, 500)
radius = 0.005
test_dataset = LightProbeDataset(test_exr_files, test_json_file, frame_dim, radius)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Number of test samples: {len(test_dataset)}")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightProbeNN().to(device)
model.load_state_dict(torch.load(model_file))
model.eval()
print("Model loaded")

# Initialize loss function
criterion = nn.MSELoss()

# Evaluate the model
total_loss = 0.0
output_images = {i: [] for i in range(test_sample_count)}
group_losses = {i: 0.0 for i in range(test_sample_count)}
group_counts = {i: 0 for i in range(test_sample_count)}

with torch.no_grad():
    for hit_point, ray_dir, rgb, group_idx in test_dataloader:
        hit_point = hit_point.to(device)
        ray_dir = ray_dir.to(device)
        rgb = rgb.to(device)
        group_idx = group_idx.numpy()

        outputs = model(hit_point, ray_dir)
        loss = criterion(outputs, rgb)
        total_loss += loss.item()

        # Collect outputs for EXR generation and calculate group losses
        for i, idx in enumerate(group_idx):
            output_images[idx].append(outputs[i].cpu().numpy())
            group_losses[idx] += criterion(outputs[i], rgb[i]).item()
            group_counts[idx] += 1

# Calculate average loss for each group
average_group_losses = {idx: group_losses[idx] / group_counts[idx] for idx in group_losses}

# Print average loss for each group
for idx, avg_loss in average_group_losses.items():
    print(f"Test Loss for group {idx}: {avg_loss:.4f}")

# Generate EXR files from model outputs
for idx, images in output_images.items():
    images = np.concatenate(images, axis=0)
    images = images.reshape((frame_dim[1], frame_dim[0], 3))  # Reshape to (500, 500, 3)
    save_exr(f'output00_{idx}.exr', images)