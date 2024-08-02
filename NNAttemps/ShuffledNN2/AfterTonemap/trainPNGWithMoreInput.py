import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

class CustomLoss(nn.Module):
    def __init__(self, threshold=0.1):
        super(CustomLoss, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, targets):
        # 计算每个样本的差值
        diff = torch.abs(outputs - targets)
        # 计算权重：差值大的部分权重大，差值小的部分权重小
        weights = torch.where(diff > self.threshold, diff, torch.zeros_like(diff))
        # 计算加权损失
        loss = torch.mean(weights * diff)
        return loss

def get_file_path(batch_folder_path, keyword, extension):
    for file_name in os.listdir(batch_folder_path):
        if keyword in file_name and file_name.endswith(extension):
            return os.path.join(batch_folder_path, file_name)
    return None

def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]
    rgb = [np.reshape(c, (size[1], size[0])) for c in rgb]

    stacked_rgb = np.stack(rgb, axis=-1)  # Keep as float32
    exr_file.close()
    return size, stacked_rgb

def read_png(file_path):
    image = Image.open(file_path).convert('RGB')
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image_np.shape[:2], image_np

class CustomDataset(Dataset):
    def __init__(self, folder_path, limit=None):
        self.hitpos_paths = []
        self.raydir_paths = []
        self.diffuse_paths = []
        self.color_paths = []

        idx = 0
        for batch_folder in os.listdir(folder_path):
            batch_folder_path = os.path.join(folder_path, batch_folder)
            hitpos_exr_path = get_file_path(batch_folder_path, "probePoses", ".exr")
            raydir_exr_path = get_file_path(batch_folder_path, "rayDirs", ".exr")
            diffuse_exr_path = get_file_path(batch_folder_path, "diffuse", ".exr")
            color_png_path = get_file_path(batch_folder_path, "ToneMapper", ".png")
            if hitpos_exr_path and raydir_exr_path and diffuse_exr_path and color_png_path:
                self.hitpos_paths.append(hitpos_exr_path)
                self.raydir_paths.append(raydir_exr_path)
                self.diffuse_paths.append(diffuse_exr_path)
                self.color_paths.append(color_png_path)
                
            if limit and idx >= limit:
                break
            idx += 1

        print(f"Loaded {len(self.hitpos_paths)} data paths from {folder_path}.")

    def __len__(self):
        return len(self.hitpos_paths)

    def __getitem__(self, idx):
        _, hitpos_image = read_exr(self.hitpos_paths[idx])
        _, raydir_image = read_exr(self.raydir_paths[idx])
        _, diffuse_image = read_exr(self.diffuse_paths[idx])
        _, color_image = read_png(self.color_paths[idx])

        input_data = np.concatenate((hitpos_image.reshape(-1, 3), raydir_image.reshape(-1, 3), 0.01 * diffuse_image.reshape(-1, 3)), axis=-1)
        color_data = color_image.reshape(-1, 3)

        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(color_data, dtype=torch.float32)

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

def train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")
    
    for batch_idx, (input_data, target) in enumerate(train_loader):
        input_data, target = input_data.to(device), target.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(input_data)
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()

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
   
    val_loss /= len(val_loader)
    return val_loss

def create_data_loaders(train_path, val_path, batch_size, train_limit=None, val_limit=None):
    train_dataset = CustomDataset(train_path, limit=train_limit)
    val_dataset = CustomDataset(val_path, limit=val_limit)
    
    num_workers = 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    batch_size = 2  # 调整批量大小，适应内存
    train_limit = 960
    val_limit = 120
    
    train_loader, val_loader = create_data_loaders(train_path, val_path, batch_size=batch_size, train_limit=train_limit, val_limit=val_limit)
    print("Data loaded")
    
    model = MLP(input_dim=9, output_dim=3).to(device)  # input_dim 更新为 9（3*3）
    # criterion = nn.MSELoss()
    criterion = CustomLoss(threshold=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    num_epochs = 500
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

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

        torch.cuda.empty_cache()
    
    print("Training complete. Testing on test dataset...")
    # test_loss = test(model, device, test_loader, criterion)
    # print(f'Test Loss: {test_loss:.4f}')

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

final_model_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model_morePNGDiffuse2.pth"
train_path = "C:/Files/CGProject/NNLightProbes/dumped_data/ShuffledData/raw/train"
val_path = "C:/Files/CGProject/NNLightProbes/dumped_data/ShuffledData/raw/val"
test_path = "C:/Files/CGProject/NNLightProbes/dumped_data/ShuffledData/raw/test"

if __name__ == '__main__':
    main()
