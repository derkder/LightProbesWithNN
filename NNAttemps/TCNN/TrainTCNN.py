import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tinycudann import NetworkWithInputEncoding, Encoding
import tinycudann as tcnn
import json

def get_exr_file_path(batch_folder_path, keyword):
    for file_name in os.listdir(batch_folder_path):
        if keyword in file_name and file_name.endswith('.exr'):
            return os.path.join(batch_folder_path, file_name)
    return None

def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, (size[1], size[0])) for c in rgb]

    stacked_rgb = np.stack(rgb, axis=-1).astype(np.float16)  # Convert to float16
    exr_file.close()
    return size, stacked_rgb

class CustomDataset(Dataset):
    def __init__(self, folder_path, limit=None):
        self.hitposes = []
        self.raydirs = []
        self.colors = []

        idx = 0
        for batch_folder in os.listdir(folder_path):
            batch_folder_path = os.path.join(folder_path, batch_folder)
            hitpos_exr_path = get_exr_file_path(batch_folder_path, "probePoses")
            raydir_exr_path = get_exr_file_path(batch_folder_path, "rayDirs")
            color_exr_path = get_exr_file_path(batch_folder_path, "output")
            if hitpos_exr_path and raydir_exr_path and color_exr_path:
                _, hitpos_image = read_exr(hitpos_exr_path)
                _, raydir_image = read_exr(raydir_exr_path)
                _, color_image = read_exr(color_exr_path)
                
                self.hitposes.append(hitpos_image.reshape(-1, 3))
                self.raydirs.append(raydir_image.reshape(-1, 3))
                self.colors.append(color_image.reshape(-1, 3))
                
            if limit and idx >= limit:
                break
            idx += 1

        self.hitposes = np.concatenate(self.hitposes, axis=0)
        self.raydirs = np.concatenate(self.raydirs, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)

    def __len__(self):
        return len(self.hitposes)

    def __getitem__(self, idx):
        input_data = np.concatenate((self.hitposes[idx], self.raydirs[idx]), axis=-1)
        color_data = self.colors[idx]

        return torch.tensor(input_data, dtype=torch.float16), torch.tensor(color_data, dtype=torch.float16)
    
class TinyCUDANNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TinyCUDANNModel, self).__init__()
        assert input_dim == 6, "Input dimension should be 6 (hitpoints and ray directions)."

        # 定义神经网络
        self.network = tcnn.NetworkWithInputEncoding(
            n_input_dims=6,  # 输入维度为6
            n_output_dims=output_dim, 
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "otype": "HashGrid",
                        "n_dims_to_encode": 3,
                        "n_levels": 8,
                        "n_features_per_level": 4,
                        "log2_hashmap_size": 19,
                        "base_resolution": 16
                    },
                    {
                        "otype": "SphericalHarmonics",
                        "n_dims_to_encode": 3,
                        "degree": 2
                    }
                ]
            },
            network_config={
                "otype": "FullyFusedMLP", 
                "activation": "ReLU", 
                "n_neurons": 64, 
                "n_hidden_layers": 2
            }
        )

    def forward(self, x):
        return self.network(x)

    def serialize(self):
        serialized_data = self.network.serialize()
        n_params = sum(p.numel() for p in self.parameters())
        params_binary = {
            "bytes": list(serialized_data),
            "subtype": None
        }
        params_type = "__half"  # 假设模型的参数类型是半精度浮点数

        model_dict = {
            "n_params": n_params,
            "params_binary": params_binary,
            "params_type": params_type
        }
        return json.dumps(model_dict)

    def deserialize(self, json_str):
        model_dict = json.loads(json_str)
        self.network.deserialize(bytes(model_dict["params_binary"]["bytes"]))

def train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")
    
    for batch_idx, (input_data, target) in enumerate(train_loader):
        # print(batch_idx)
        input_data, target = input_data.to(device), target.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_data)
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        torch.cuda.empty_cache()

    train_loss /= len(train_loader)
    return train_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_data, target in val_loader:
            input_data, target = input_data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(input_data)
                loss = criterion(output, target)
            val_loss += loss.item()
            torch.cuda.empty_cache()
   
    val_loss /= len(val_loader)
    return val_loss

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for input_data, target in test_loader:
            input_data, target = input_data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(input_data)
                test_loss += criterion(output, target).item()
            torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    return test_loss

def create_data_loaders(train_path, val_path, test_path, batch_size, train_limit=None, val_limit=None, test_limit=None):
    train_dataset = CustomDataset(train_path, limit=train_limit)
    val_dataset = CustomDataset(val_path, limit=val_limit)
    test_dataset = CustomDataset(test_path, limit=test_limit)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    batch_size =  104976
    train_limit = 80
    val_limit = 10
    test_limit = 10
    
    train_loader, val_loader, test_loader = create_data_loaders(train_path, val_path, test_path, batch_size=batch_size, train_limit=train_limit, val_limit=val_limit, test_limit=test_limit)
    print("Data loaded")
    
    model = TinyCUDANNModel(input_dim=6, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    num_epochs = 500
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    writer = SummaryWriter(log_dir='runs/experiment_1')

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4)
        val_loss = validate(model, device, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), final_model_path)
            model_json = model.serialize()
            with open("NNAttemps/ShuffledNN2/models/final_light_probe_model_tcnn.json", "w") as f:
                f.write(model_json)

        torch.cuda.empty_cache()
    
    print("Training complete. Testing on test dataset...")
    test_loss = test(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')
    writer.add_scalar('Loss/test', test_loss, epoch)

    writer.close()

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

final_model_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model_tcnn.pth"
train_path = "C:/Files/CGProject/NNLightProbes/dumped_data/AllData/rawraw/ValidData/train"
val_path = "C:/Files/CGProject/NNLightProbes/dumped_data/AllData/rawraw/ValidData/val"
test_path = "C:/Files/CGProject/NNLightProbes/dumped_data/AllData/rawraw/ValidData/test"

if __name__ == '__main__':
    main()