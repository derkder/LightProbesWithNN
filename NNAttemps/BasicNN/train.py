import json
import numpy as np
import OpenEXR
import Imath
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

 
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

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
    save_exr('stacked_rgb.exr', stacked_rgb)
    # Crop to the first 500 rows and 500 columns
    cropped_rgb = stacked_rgb[:500, :500, :]
    save_exr('cropped_rgb.exr', cropped_rgb)
    
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
                data.append((hit_points[j], ray_dirs[j], rgb_values[j // self.frame_dim[1], j % self.frame_dim[0]]))
            print(f"Processed {i+1}/{len(self.exr_files)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hit_point, ray_dir, rgb = self.data[idx]
        return torch.tensor(hit_point, dtype=torch.float32), torch.tensor(ray_dir, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32)
    
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
sample_count = 10
exr_files = [f"D:/Projects/LightProbesWithNN/dumped_data/dim500/train/frame_{i:04d}/Mogwai.AccumulatePass.output.100000.exr" for i in range(sample_count)]
json_file = "D:/Projects/LightProbesWithNN/dumped_data/dim500/train/info.json"
# exr_files = [f"C:/Files/CGProject/NNLightProbes/dumped_data/dim500/all/train/frame_{i:04d}/Mogwai.AccumulatePass.output.100000.exr" for i in range(sample_count)]
# json_file = "C:/Files/CGProject/NNLightProbes/dumped_data/dim500/all/train/info.json"
frame_dim = (500, 500)
radius = 0.005
dataset = LightProbeDataset(exr_files, json_file, frame_dim, radius)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Number of samples: {len(dataset)}")

# Initialize the network, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightProbeNN().to(device)
print(f"Using device: {device}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
print("Network Initialized")

losses = []
best_loss = float('inf')
loss_model_path = "NNAttemps/BasicNN/models/best_light_probe_model3.pth"
final_model_path = "NNAttemps/BasicNN/models/final_light_probe_model3.pth"
# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    for hit_point, ray_dir, rgb in dataloader:
        hit_point = hit_point.to(device)
        ray_dir = ray_dir.to(device)
        rgb = rgb.to(device)

        optimizer.zero_grad()
        outputs = model(hit_point, ray_dir)
        loss = criterion(outputs, rgb)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    losses.append(loss.item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), loss_model_path)
        print(f"Saved better model with Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), final_model_path)

# plot the loss
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.show()