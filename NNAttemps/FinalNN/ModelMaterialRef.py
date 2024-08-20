import numpy as np
import torch
import torch.nn as nn
import OpenEXR
import Imath

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

# Function to generate primaryRayOrigin, rayDir, and materials
def generate_rays(frame_dim, radius, kProbeLoc):
    hit_points = []
    ray_dirs = []
    normals = []
    materials = []
    for y in range(frame_dim[1]):
        for x in range(frame_dim[0]):
            theta = np.pi * y / frame_dim[1]
            phi = 2 * np.pi * x / frame_dim[0]
            x_val = radius * np.sin(theta) * np.cos(phi)
            z_val = radius * np.sin(theta) * np.sin(phi)
            y_val = radius * np.cos(theta)
            ray_dir = np.array([x_val, y_val, z_val])
            ray_dir = - ray_dir / np.linalg.norm(ray_dir)
            normal = -ray_dir
            hit_points.append(kProbeLoc)
            ray_dirs.append(ray_dir)
            normals.append(normal)
            materials.append([0.0, 1.0])  # Add material attributes (float2)
    return np.array(hit_points), np.array(ray_dirs), np.array(normals), np.array(materials)

# Load Data
kprobeloc = np.array([0.16089, 0.28183, 0.08310])
# kprobeloc = np.array([-0.240646, 0.020569, 0.082068])
# kprobeloc = np.array([-0.085158, 0.4904897, -0.12996])

model_file = "NNAttemps/FinalNN/models/final_light_probe_model_MATERIALPNGBETTER.pth"
output_path = "NNAttemps/FinalNN/output/output00_77.exr"
frame_dim = (1920, 1080)
radius = 0.005

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(11, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, 512) # 增加神经元数量
        self.fc3 = nn.Linear(512, 256) 
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_dim) # 增加层数
        self.dropout = nn.Dropout(p=0.5)      # 添加Dropout层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
def main():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=11,  output_dim=3).to(device)  # input_dim now 11 (9 + 2 for materials)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Generate rays
    hit_points, ray_dirs, normals, materials = generate_rays(frame_dim, radius, kprobeloc)

    # Flatten the rays for batch processing
    hit_points_flat = hit_points.reshape(-1, 3)
    ray_dirs_flat = ray_dirs.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)
    materials_flat = materials.reshape(-1, 2)  # Flatten materials

    # Prepare input data for the model
    input_data = np.concatenate((hit_points_flat, ray_dirs_flat, normals_flat, materials_flat), axis=1)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    # Process with the model
    batch_size = 1024  # Adjust batch size based on your GPU memory
    output_img = []
    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch_input = input_tensor[i:i+batch_size]
            output = model(batch_input)
            output_img.append(output.cpu().numpy())

    # Combine the output batches and reshape to image dimensions
    output_img = np.concatenate(output_img, axis=0)
    output_img = output_img.reshape((frame_dim[1], frame_dim[0], 3))

    # Save the result
    save_exr(output_path, output_img)

if __name__ == '__main__':
    main()
