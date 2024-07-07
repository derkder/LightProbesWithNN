import json
import numpy as np
import OpenEXR
import Imath
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# Function to read EXR file
def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, size) for c in rgb]
    return np.stack(rgb, axis=-1)

# Function to generate primaryRayOrigin and rayDir
def generate_rays(frame_dim, radius, kProbeLoc):
    primary_ray_origins = []
    ray_dirs = []
    for y in range(frame_dim[1]):
        for x in range(frame_dim[0]):
            theta = np.pi * y / frame_dim[1]
            phi = 2 * np.pi * x / frame_dim[0]
            x_val = radius * np.sin(theta) * np.cos(phi)
            z_val = radius * np.sin(theta) * np.sin(phi)
            y_val = radius * np.cos(theta)
            primary_ray_origin = np.array([x_val, y_val, z_val]) + kProbeLoc
            ray_dir = np.array([x_val, y_val, z_val])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            primary_ray_origins.append(primary_ray_origin)
            ray_dirs.append(ray_dir)
    return np.array(primary_ray_origins), np.array(ray_dirs)

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
            primary_ray_origins, ray_dirs = generate_rays(self.frame_dim, self.radius, kProbeLoc)
            for j in range(self.frame_dim[0] * self.frame_dim[1]):
                data.append((primary_ray_origins[j], ray_dirs[j], rgb_values[j // self.frame_dim[1], j % self.frame_dim[0]]))
            print(f"Processed {i+1}/{len(self.exr_files)}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        primary_ray_origin, ray_dir, rgb = self.data[idx]
        return torch.tensor(primary_ray_origin, dtype=torch.float32), torch.tensor(ray_dir, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32)
    
class LightProbeNN(nn.Module):
    def __init__(self):
        super(LightProbeNN, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, primary_ray_origin, ray_dir):
        x = torch.cat((primary_ray_origin, ray_dir), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x




# Load Data
sample_count = 5
exr_files = [f"D:/Projects/LightProbesWithNN/dumped_data/frame_{i:04d}/Mogwai.AccumulatePass.output.3000.exr" for i in range(sample_count)]
json_file = "D:/Projects/LightProbesWithNN/dumped_data/info.json"
frame_dim = (1000, 1000)
radius = 0.005
dataset = LightProbeDataset(exr_files, json_file, frame_dim, radius)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Number of samples: {len(dataset)}")

# Initialize the network, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightProbeNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Network Initialized")


best_loss = float('inf')
loss_model_path = "NNAttemps/BasicNN/models/best_light_probe_model.pth"
final_model_path = "NNAttemps/BasicNN/models/best_light_probe_model.pth"
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for primary_ray_origin, ray_dir, rgb in dataloader:
        primary_ray_origin = primary_ray_origin.to(device)
        ray_dir = ray_dir.to(device)
        rgb = rgb.to(device)

        optimizer.zero_grad()
        outputs = model(primary_ray_origin, ray_dir)
        loss = criterion(outputs, rgb)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), loss_model_path)
        print(f"Saved better model with Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), final_model_path)



# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()  # 确保模型处于训练模式
#     total_loss = 0
#     count = 0
#     for primary_ray_origin, ray_dir, rgb in dataloader:
#         primary_ray_origin = primary_ray_origin.to(device)
#         ray_dir = ray_dir.to(device)
#         rgb = rgb.to(device)

#         optimizer.zero_grad()
#         outputs = model(primary_ray_origin, ray_dir)
#         loss = criterion(outputs, rgb)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         count += 1

#     average_loss = total_loss / count
#     print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}")

#     # 如果当前训练损失是最低的，保存模型
#     if average_loss < best_loss:
#         best_loss = average_loss
#         torch.save(model.state_dict(), model_path)
#         print(f"Saved better model with Training Loss: {average_loss:.4f}")