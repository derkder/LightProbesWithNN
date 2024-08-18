import numpy as np
import torch
import torch.nn as nn
import OpenEXR
import Imath
import os
import cv2
from PIL import Image


def linear_to_srgb(image):
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * (image ** (1/2.4)) - 0.055)


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

def convert_png_to_exr(png_path, exr_path):
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise Exception(f"Failed to load image from {png_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_linear = np.power(image_rgb / 255.0, 2.2)
    save_exr(exr_path, image_linear)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 128)
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
    model.load_state_dict(torch.load("NNAttemps/FinalNN/models/final_light_probe_model_NORMALPNG.pth"))
    model.eval()

    # 生成光线model_file = "NNAttemps/ShuffledNN2/models/colab/final_light_probe_model_NORMALPNG (5).pth"
    frame_dim = (1920, 1080)

    # 加载和预处理EXR文件
    batch_folder_path = "C:/Files/CGProject/NNLightProbes/dumped_data/TestData/raw/frame_0001"
    exr_paths = {
        "hitposes": os.path.join(batch_folder_path, "Mogwai.NetworkPass.hitposes.3000.exr"),
        "raydirs": os.path.join(batch_folder_path, "Mogwai.NetworkPass.raydirs.3000.exr"),
        "normals": os.path.join(batch_folder_path, "Mogwai.NetworkPass.normals1.3000.exr"),
    }
    
    images = {}
    for key, path in exr_paths.items():
        _, image = read_exr(path)
        images[key] = image.reshape(-1, 3)

    # 生成mask，基于raydirs图像中不为0的像素
    mask = np.any(images["raydirs"] != 0, axis=-1)
    mask = mask.reshape((frame_dim[1], frame_dim[0]))  # 重新调整mask的形状，使其与target_img形状一致

    # Prepare input data
    input_data = np.concatenate(
        [images["hitposes"], images["raydirs"], images["normals"]], 
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
    # output_img = srgb_to_linear(output_img)

    # 读取并转换目标PNG文件为EXR
    target_png_path = os.path.join(batch_folder_path, "Mogwai.ToneMapper.dst.3000.png")  # 请替换为你的目标PNG文件路径
    target_exr_path = os.path.join(batch_folder_path, "Mogwai.ToneMapper.dst.3000.exr")
    convert_png_to_exr(target_png_path, target_exr_path)  # 将PNG转换为EXR
    _, target_img = read_exr(target_exr_path)

    # 应用mask，仅保留有颜色的部分
    target_img[mask] = output_img[mask]

    # 保存最终结果为EXR格式 
    save_exr("NNAttemps/FinalNN/output/output00_90_masked.exr", target_img)

    # 生成仅包含mask部分的图像并保存为EXR格式
    mask_only_img = np.zeros_like(target_img)
    mask_only_img[mask] = output_img[mask]
    save_exr("NNAttemps/FinalNN/output/output00_90_mask_only.exr", mask_only_img)

if __name__ == '__main__':
    main()
