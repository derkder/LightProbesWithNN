import numpy as np
import torch
import torch.nn as nn
import OpenEXR
import Imath
import os
from PIL import Image

# Function to convert sRGB to linear color space
def srgb_to_linear(image):
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

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

# Function to convert PNG to EXR
def png_to_exr(png_path, exr_path):
    image = Image.open(png_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to 0-1
    save_exr(exr_path, image_np)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(6, 128)
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
    model = MLP(input_dim=6, output_dim=3).to(device)
    model.load_state_dict(torch.load("NNAttemps/ShuffledNN2/models/final_light_probe_model_morePNG.pth"))
    model.eval()

    # 生成光线
    frame_dim = (1920, 1080)

    # 加载和预处理EXR文件
    batch_folder_path = "C:/Files/CGProject/NNLightProbes/dumped_data/TestData/raw/frame_0000"
    exr_paths = {
        "hitposes": os.path.join(batch_folder_path, "Mogwai.NetworkPass.hitposes.3000.exr"),
        "raydirs": os.path.join(batch_folder_path, "Mogwai.NetworkPass.raydirs.3000.exr")
    }
    
    images = {}
    for key, path in exr_paths.items():
        _, image = read_exr(path)
        images[key] = image.reshape(-1, 3)

    # images["raydirs"] *= -1.0
    # Prepare input data
    input_data = np.concatenate(
        [images["hitposes"], images["raydirs"]], 
        axis=1
    ).reshape(-1, 6)
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

    # 读取并转换目标PNG文件为EXR
    target_png_path = os.path.join(batch_folder_path, "Mogwai.ToneMapper.dst.3000.png")  # 请替换为你的目标PNG文件路径
    target_exr_path = os.path.join(batch_folder_path, "Mogwai.ToneMapper.dst.3000.exr")
    png_to_exr(target_png_path, target_exr_path)  # 将PNG转换为EXR
    _, target_img = read_exr(target_exr_path)

    # 应用mask，仅保留有颜色的部分
    mask = np.any(np.abs(output_img - 0) >= 0.1, axis=-1)
    target_img[mask] = output_img[mask]

    # 保存最终结果为EXR格式
    target_img = srgb_to_linear(target_img)
    save_exr("NNAttemps/ShuffledNN2/out_imgs/output00_90_masked.exr", target_img)

    # 生成仅包含mask部分的图像并保存为EXR格式
    mask_only_img = np.zeros_like(target_img)
    mask_only_img[mask] = output_img[mask]
    save_exr("NNAttemps/ShuffledNN2/out_imgs/output00_90_mask_only.exr", mask_only_img)

if __name__ == '__main__':
    main()
