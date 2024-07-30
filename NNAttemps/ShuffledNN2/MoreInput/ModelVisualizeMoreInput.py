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
    
    stacked_rgb = np.stack(rgb, axis=-1).astype(np.float16)
    exr_file.close()
    return size, stacked_rgb

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=15, hidden_dim=128, output_dim=3).to(device).half()  # 模型转换为float16
    model.load_state_dict(torch.load("NNAttemps/ShuffledNN2/models/final_light_probe_model_moreinput.pth"))
    model.eval()

    # 生成光线
    kprobeloc = np.array([0.16089, 0.28183, 0.08310])
    frame_dim = (1920, 1080)
    radius = 0.005
    hit_points, ray_dirs = generate_rays(frame_dim, radius, kprobeloc)

    # 加载和预处理EXR文件
    batch_folder_path = "C:/Files/CGProject/NNLightProbes/dumped_data/tempp/frame_0001"
    exr_paths = {
        "diffuse": os.path.join(batch_folder_path, "Mogwai.CollectData.diffuse.4001.exr"),
        "roughnessemmisive": os.path.join(batch_folder_path, "Mogwai.CollectData.roughnessemmisive.4001.exr"),
        "specular": os.path.join(batch_folder_path, "Mogwai.CollectData.specular.4001.exr"),
        "output": os.path.join(batch_folder_path, "Mogwai.AccumulatePass.output.4001.exr"),
    }
    
    images = {}
    for key, path in exr_paths.items():
        _, image = read_exr(path)
        images[key] = image.reshape(-1, 3)

    # Apply scaling to specific inputs
    # images["diffuse"] *= 0.01
    # images["roughnessemmisive"] *= 0.1
    # images["specular"] *= 0.1
    images["diffuse"] *= 0.1

    # Prepare input data
    input_data = np.concatenate(
        [hit_points, ray_dirs, images["diffuse"], images["roughnessemmisive"], images["specular"]], 
        axis=1
    ).reshape(-1, 15)
    input_tensor = torch.tensor(input_data, dtype=torch.float16).to(device)  # 输入数据转换为float16

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
