import numpy as np
import torch
import torch.nn as nn
import OpenEXR
import Imath

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
    return size, stacked_rgb

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
            hit_point = np.array([x_val, y_val, z_val]) + kProbeLoc
            ray_dir = np.array([x_val, y_val, z_val])
            ray_dir = - ray_dir / np.linalg.norm(ray_dir)
            hit_points.append(hit_point)
            ray_dirs.append(ray_dir)
    return np.array(hit_points), np.array(ray_dirs)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Load Data
kprobeloc = np.array([0.24709033995380147, 0.08912807584807496, 0.23231801106817485])
model_file = "NNAttemps/ShuffledNNMoreInput/models/final_light_probe_model_6.pth"
output_path = "NNAttemps/ShuffledNNMoreInput/out_imgs/output00_6.exr"
frame_dim = (1920, 1080)
radius = 0.005

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=15, hidden_dim=256, output_dim=3).to(device)
model.load_state_dict(torch.load(model_file))
model.eval()

# Generate rays
hit_points, ray_dirs = generate_rays(frame_dim, radius, kprobeloc)

# Generate dummy data for diffuse, roughEmmi, and specular
diffuse_file = "NNAttemps/testData/Mogwai.ReflectMapGen.diffuse.4999.exr"
roughnessemmi_file = "NNAttemps/testData/Mogwai.ReflectMapGen.roughnessemmisive.4999.exr"
specular_file = "NNAttemps/testData/Mogwai.ReflectMapGen.specular.4999.exr"
_, diffuse = read_exr(diffuse_file)
_, roughEmmi = read_exr(roughnessemmi_file)
_, specular = read_exr(specular_file)

# Flatten the rays and additional data for batch processing
hit_points_flat = hit_points.reshape(-1, 3)
ray_dirs_flat = ray_dirs.reshape(-1, 3)
diffuse_flat = diffuse.reshape(-1, 3)
roughEmmi_flat = roughEmmi.reshape(-1, 3)
specular_flat = specular.reshape(-1, 3)

# Prepare input data for the model
input_data = np.concatenate((hit_points_flat, ray_dirs_flat, diffuse_flat, roughEmmi_flat, specular_flat), axis=1)
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
