import OpenEXR
import Imath
import numpy as np
import cv2

normal_path = "C:/Files/CGProject/NNLightProbes/dumped_data/TestData/raw/frame_0001/Mogwai.NetworkPass.normals.3000.exr"
smooth_path = "C:/Files/CGProject/NNLightProbes/dumped_data/TestData/raw/frame_0001/Mogwai.NetworkPass.normals1.3000.exr"

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

# Function to smooth normals with multiple passes
def smooth_normals(normals, kernel_size=7, passes=5):
    smoothed_normals = normals.copy()
    
    for _ in range(passes):
        smoothed_normals = cv2.GaussianBlur(smoothed_normals, (kernel_size, kernel_size), 0)
    
    # Normalize the normals
    norm = np.linalg.norm(smoothed_normals, axis=-1, keepdims=True)
    smoothed_normals /= np.clip(norm, 1e-8, None)
    
    return smoothed_normals

# Read the EXR image
file_path = normal_path
size, normals = read_exr(file_path)

# Smooth the normals
smoothed_normals = smooth_normals(normals)

# Convert the smoothed normals back to EXR and save
def write_exr(file_path, size, data):
    exr_file = OpenEXR.OutputFile(file_path, OpenEXR.Header(size[0], size[1]))
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [data[:, :, i].astype(np.float32).tobytes() for i in range(3)]
    exr_file.writePixels({'R': channels[0], 'G': channels[1], 'B': channels[2]})
    exr_file.close()

output_file_path = smooth_path
write_exr(output_file_path, size, smoothed_normals)
