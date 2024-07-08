import torch
import torch.nn as nn
import numpy as np
import OpenEXR
import Imath
import json
from torch.utils.data import Dataset, DataLoader

# 定义读取EXR文件的函数
def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, size) for c in rgb]
    return np.stack(rgb, axis=-1)

# 定义生成光线的函数
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

# 定义自定义数据集
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

# 定义神经网络模型
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

# 加载测试数据
sample_count = 2
test_exr_files = [f"D:/Projects/LightProbesWithNN/dumped_data/test/frame_{i:04d}/Mogwai.AccumulatePass.output.3000.exr" for i in range(sample_count)]
test_json_file = "D:/Projects/LightProbesWithNN/dumped_data/test/info.json"
test_frame_dim = (1000, 1000)  # 替换为你的测试帧尺寸
test_radius = 0.005  # 替换为你的测试半径
test_dataset = LightProbeDataset(test_exr_files, test_json_file, test_frame_dim, test_radius)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Number of test samples: {len(test_dataset)}")

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightProbeNN().to(device)
model.load_state_dict(torch.load("NNAttemps/BasicNN/models/best_light_probe_model.pth"))
model.eval()  # 设置模型为评估模式
print("Model loaded and set to evaluation mode")

# 进行推理
with torch.no_grad():  # 在推理时不需要计算梯度
    for primary_ray_origin, ray_dir, rgb in test_dataloader:
        primary_ray_origin = primary_ray_origin.to(device)
        ray_dir = ray_dir.to(device)
        rgb = rgb.to(device)

        outputs = model(primary_ray_origin, ray_dir)
        
        # 这里可以根据需要处理输出，例如计算误差、保存结果等
        # 例如，计算与真实RGB值的均方误差
        mse_loss = nn.MSELoss()(outputs, rgb)
        print(f"Test Loss: {mse_loss.item():.4f}")

        # 如果需要保存输出结果，可以将其转换为numpy数组并保存
        outputs_np = outputs.cpu().numpy()
        # 保存或处理outputs_np...

print("Inference completed")