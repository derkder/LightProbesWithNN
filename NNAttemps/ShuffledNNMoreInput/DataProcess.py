import os
import OpenEXR
import Imath
import numpy as np
import random
import ctypes
import json

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

def read_cur_seed_loc(filename, idx):
    with open(filename, 'r') as f:
        data = json.load(f)
        for entry in data:
            if entry['idx'] == idx:
                kProbeLoc = np.array([entry['new_x'], entry['new_y'], entry['new_z']])
                print(entry['curSeed'], kProbeLoc)
                return entry['curSeed'], kProbeLoc
        return None  # Handle case where idx is not found

# Function to generate primaryRayOrigin and rayDir
def generate_ray(frame_dim, radius, kProbeLoc, x, y):
    theta = np.pi * y / frame_dim[1]
    phi = 2 * np.pi * x / frame_dim[0]
    x_val = radius * np.sin(theta) * np.cos(phi)
    z_val = radius * np.sin(theta) * np.sin(phi)
    y_val = radius * np.cos(theta)
    hit_point = np.array([x_val, y_val, z_val]) + kProbeLoc
    ray_dir = np.array([x_val, y_val, z_val])
    ray_dir = - ray_dir / np.linalg.norm(ray_dir)
    return hit_point, ray_dir

def should_return_true(x, y, seed):
    total_calls = 1920 * 1080
    max_true_count = 10000

    h = (ctypes.c_int32(x * 374761393).value +
         ctypes.c_int32(y * 668265263).value +
         ctypes.c_int32(seed * 1013904223).value)
    h = ctypes.c_int32(h).value
    h = ctypes.c_int32((h ^ (h >> 13)) * 1274126177).value
    h = ctypes.c_int32(h).value
    h = ctypes.c_int32(h ^ (h >> 16)).value
    random_value = abs(h) % total_calls

    return random_value < max_true_count

# 函数：从文件夹中读取数据x
def read_data_from_folders(input_path, json_path, start_idx, exclude_indices):
    print(f"startidx{start_idx}")
    data = []
    
    for i in range(start_idx, start_idx + total_process_num):
        if i in exclude_indices:
            continue
        # else:
        #     file_name_format = "frame_{:04d}/Mogwai.AccumulatePass.output.50000.exr".format(i)
        #     image_path = f"{input_path}/{file_name_format}"
        #     print(image_path)
        file_name_format = "frame_{:04d}/Mogwai.AccumulatePass.output.4000.exr".format(i)
        image_path = f"{input_path}/{file_name_format}"
        file_name_format = "frame_{:04d}/Mogwai.ReflectMapGen.diffuse.4000.exr".format(i)
        diffuse_path = f"{input_path}/{file_name_format}"
        file_name_format = "frame_{:04d}/Mogwai.ReflectMapGen.roughnessemmisive.4000.exr".format(i)
        roughEmmi_path = f"{input_path}/{file_name_format}"
        file_name_format = "frame_{:04d}/Mogwai.ReflectMapGen.specular.4000.exr".format(i)
        specular_path = f"{input_path}/{file_name_format}"
        
        if os.path.exists(image_path):
            # 读取EXR图片    
            size, rgb_values = read_exr(image_path)
            _, diffuse_values = read_exr(diffuse_path)
            _, roughEmmi_values = read_exr(roughEmmi_path)
            _, specular_values = read_exr(specular_path)
            # print(f"size{size}") # 1920 1080
            seed, probeLoc = read_cur_seed_loc(json_path, i)
            if seed is None:
                print("idx error")
                continue
            for y in range(size[1]):
                for x in range(size[0]):
                    if should_return_true(x, y, seed):
                        hit_point, ray_dir = generate_ray(frame_dim, radius, probeLoc, x, y)
                        data.append((rgb_values[y, x], hit_point, ray_dir, diffuse_values[y, x], roughEmmi_values[y, x], specular_values[y, x]))
        print(len(data))
        print(f"Read Processed {i}/{start_idx + total_process_num}")
    
    return data

# 函数：打乱数据
def shuffle_data(data):
    random.shuffle(data)
    return data

# 函数：分割数据集
def split_data(data, train_ratio=0.8, val_ratio=0.1):
    num_samples = len(data)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    
    train_data = data[:num_train]
    val_data = data[num_train:num_train+num_val]
    test_data = data[num_train+num_val:]
    
    return train_data, val_data, test_data

# 函数：保存数据到文件夹
def save_data_to_folders(data, output_path, batch_dim=(100, 100)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    batch_size = batch_dim[0] * batch_dim[1]  # 每组数据点的数量
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        
        # 创建文件夹
        batch_folder = os.path.join(output_path, f"batch_{batch_idx}")
        os.makedirs(batch_folder, exist_ok=True)
        
        # 保存每个数据点的颜色数据为EXR格式
        color_image = np.zeros((batch_dim[0], batch_dim[1], 3), dtype=np.float32)
        diffuse_image = np.zeros((batch_dim[0], batch_dim[1], 3), dtype=np.float32)
        roughEmmi_image = np.zeros((batch_dim[0], batch_dim[1], 3), dtype=np.float32)
        specular_image = np.zeros((batch_dim[0], batch_dim[1], 3), dtype=np.float32)
        for i, (color, _, _, diffuse, roughEmmi, specular) in enumerate(batch_data):
            x = i % batch_dim[0]
            y = i // batch_dim[1]
            color_image[y, x] = color
            diffuse_image[y, x] = diffuse
            roughEmmi_image[y, x] = roughEmmi
            specular_image[y, x] = specular

        
        exr_header = OpenEXR.Header(batch_dim[0], batch_dim[1])
        exr_header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                                  'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                                  'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
         
        exr_file_c = OpenEXR.OutputFile(os.path.join(batch_folder, f"color.exr"), exr_header)
        exr_file_c.writePixels({'R': color_image[:,:,0].astype(np.float32).tobytes(),
                              'G': color_image[:,:,1].astype(np.float32).tobytes(),
                              'B': color_image[:,:,2].astype(np.float32).tobytes()})
        exr_file_d = OpenEXR.OutputFile(os.path.join(batch_folder, f"diffuse.exr"), exr_header)
        exr_file_d.writePixels({'R': diffuse_image[:,:,0].astype(np.float32).tobytes(),
                              'G': diffuse_image[:,:,1].astype(np.float32).tobytes(),
                              'B': diffuse_image[:,:,2].astype(np.float32).tobytes()})
        exr_file_r = OpenEXR.OutputFile(os.path.join(batch_folder, f"roughEmmi.exr"), exr_header)
        exr_file_r.writePixels({'R': roughEmmi_image[:,:,0].astype(np.float32).tobytes(),
                              'G': roughEmmi_image[:,:,1].astype(np.float32).tobytes(),
                              'B': roughEmmi_image[:,:,2].astype(np.float32).tobytes()})
        exr_file_s = OpenEXR.OutputFile(os.path.join(batch_folder, f"specular.exr"), exr_header)
        exr_file_s.writePixels({'R': specular_image[:,:,0].astype(np.float32).tobytes(),
                              'G': specular_image[:,:,1].astype(np.float32).tobytes(),
                              'B': specular_image[:,:,2].astype(np.float32).tobytes()})
       
        del exr_file_c, exr_file_d, exr_file_r, exr_file_s
        
        # 保存hitpoint和raydir数据为JSON格式
        hitpoint_data = []
        raydir_data = []
        for _, hit_point, ray_dir, _, _, _ in batch_data:
            hitpoint_data.append(hit_point.tolist())
            raydir_data.append(ray_dir.tolist())
        
        json_data = {
            "num": len(batch_data),
            "hitpoints": hitpoint_data,
            "raydir": raydir_data
        }
        
        with open(os.path.join(batch_folder, f"data.json"), 'w') as f:
            json.dump(json_data, f)
        
        print(f"Write Processed {batch_idx}/{num_batches}")

# 示例调用
json_path =  "C:/Files/CGProject/NNLightProbes/dumped_data/tempFullData/raw/info.json"
input_path = "C:/Files/CGProject/NNLightProbes/dumped_data/tempFullData/raw/"  # 输入文件夹路径
output_path = "C:/Files/CGProject/NNLightProbes/dumped_data/tempFullData/processed_real/"  # 输出文件夹路径
# json_path =  "D:/Projects/LightProbesWithNN/dumped_data/temptemp/raw/info.json"
# input_path = "D:/Projects/LightProbesWithNN/dumped_data/temptemp/raw/"  # 输入文件夹路径
# output_path = "D:/Projects/LightProbesWithNN/dumped_data/temptemp/processed_real/"  # 输出文件夹路径
frame_dim = (1920, 1080)
radius = 0.005
total_process_num = 40
start_idx = 0  # 从这个索引开始处理
exclude_indices = [1, 11]  # 要排除的索引数组

# 读取数据
data = read_data_from_folders(input_path, json_path, start_idx, exclude_indices)

# 打乱数据
shuffled_data = shuffle_data(data)

# 分割数据集
train_data, val_data, test_data = split_data(shuffled_data)

# 保存数据到图片
save_data_to_folders(train_data, os.path.join(output_path, "train"))
save_data_to_folders(val_data, os.path.join(output_path, "val"))
save_data_to_folders(test_data, os.path.join(output_path, "test"))

