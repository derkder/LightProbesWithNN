import numpy as np
import torch
import torch.nn as nn
import OpenEXR
import Imath
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
            ray_dir = np.array([x_val, y_val, z_val])
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            hit_points.append(kProbeLoc)
            ray_dirs.append(ray_dir)
    return np.array(hit_points), np.array(ray_dirs)

# Function to read EXR file
def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, (size[1], size[0])) for c in rgb]
    
    stacked_rgb = np.stack(rgb, axis=-1).astype(np.float32)
    exr_file.close()
    return size, stacked_rgb

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=9, output_dim=3).to(device)
    
    # 尝试加载模型参数，忽略尺寸不匹配的层
    try:
        state_dict = torch.load("NNAttemps/ShuffledNN2/models/final_light_probe_model_PNGDiffuseWeightedLoss.pth")
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
    
    model.eval()

    # 生成光线
    kprobeloc = np.array([0.16089, 0.28183, 0.08310])
    # kprobeloc = np.array([-0.085158, 0.4904897, -0.12996])
    # kprobeloc = np.array([-0.240646, 0.020569, 0.082068])
    frame_dim = (1920, 1080)
    radius = 0.005
    hit_points, ray_dirs = generate_rays(frame_dim, radius, kprobeloc)

    # 加载和预处理EXR文件
    batch_folder_path = "C:/Files/CGProject/NNLightProbes/dumped_data/tempp/A_SET"
    exr_paths = {
        "diffuse": os.path.join(batch_folder_path, "Mogwai.CollectData.diffuse.4001.exr")
    }
    
    images = {}
    for key, path in exr_paths.items():
        _, image = read_exr(path)
        images[key] = image.reshape(-1, 3)

    # Apply scaling to specific inputs
    images["diffuse"] *= 0.01

    # Prepare input data
    input_data = np.concatenate(
        [hit_points, ray_dirs, images["diffuse"]], 
        axis=1
    ).reshape(-1, 9)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    # 处理模型输出
    batch_size = 1024
    output_img = []
    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch_input = input_tensor[i:i+batch_size]
            output = model(batch_input)
            output_img.append(output.cpu().numpy())

    # 合并输出并重塑为图像维度
    output_img = np.concatenate(output_img, axis=0)
    output_img = output_img.reshape((frame_dim[1], frame_dim[0], 3))

    # 保存结果
    save_exr("NNAttemps/ShuffledNN2/out_imgs/output00_9.exr", output_img)

if __name__ == '__main__':
    main()
