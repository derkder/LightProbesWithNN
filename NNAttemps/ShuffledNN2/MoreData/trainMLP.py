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

    stacked_rgb = np.stack(rgb, axis=-1).astype(np.float16)  # Convert to float16
    exr_file.close()
    return size, stacked_rgb

def read_png(file_path):
    image = Image.open(file_path).convert('RGB')
    image_np = np.array(image).astype(np.float16) / 255.0  # Normalize to [0, 1]
    return image_np.shape[:2], image_np

class CustomDataset(Dataset):
    def __init__(self, folder_path, limit=None):
        self.hitposes = []
        self.raydirs = []
        self.colors = []

        idx = 0
        for batch_folder in os.listdir(folder_path):
            batch_folder_path = os.path.join(folder_path, batch_folder)
            hitpos_exr_path = get_file_path(batch_folder_path, "probePoses", ".exr")
            raydir_exr_path = get_file_path(batch_folder_path, "rayDirs", ".exr")
            color_png_path = get_file_path(batch_folder_path, "ToneMapper", ".png")
            if hitpos_exr_path and raydir_exr_path and color_png_path:
                _, hitpos_image = read_exr(hitpos_exr_path)
                _, raydir_image = read_exr(raydir_exr_path)
                _, color_image = read_png(color_png_path)
                
                self.hitposes.append(hitpos_image.reshape(-1, 3))
                self.raydirs.append(raydir_image.reshape(-1, 3))
                self.colors.append(color_image.reshape(-1, 3))
                
            if limit and idx >= limit:
                break
            idx += 1

        self.hitposes = np.concatenate(self.hitposes, axis=0)
        self.raydirs = np.concatenate(self.raydirs, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)

        print(f"Loaded {len(self.hitposes)} data points from {folder_path}.")

    def __len__(self):
        return len(self.hitposes)

    def __getitem__(self, idx):
        input_data = np.concatenate((self.hitposes[idx], self.raydirs[idx]), axis=-1)
        color_data = self.colors[idx]

        return torch.tensor(input_data, dtype=torch.float16), torch.tensor(color_data, dtype=torch.float16)

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

def train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")
    
    for batch_idx, (input_data, target) in enumerate(train_loader):
        input_data, target = input_data.to(device), target.to(device)
        
        # optimizer.zero_grad() 在每个batch开始时调用
        with torch.cuda.amp.autocast():
            outputs = model(input_data)
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        
        # 梯度累加步骤
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        # torch.cuda.empty_cache() 这行可以去掉，它会影响性能

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
            # torch.cuda.empty_cache() 这行可以去掉，它会影响性能
   
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
            # torch.cuda.empty_cache() 这行可以去掉，它会影响性能
    test_loss /= len(test_loader)
    return test_loss

def create_data_loaders(train_path, val_path, batch_size, train_limit=None, val_limit=None):
    train_dataset = CustomDataset(train_path, limit=train_limit)
    val_dataset = CustomDataset(val_path, limit=val_limit)
    
    num_workers = 4  # 根据硬件配置调整
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    # batch_size =  524288 # 2 ^ 10 
    batch_size = 1048576 # 2 ^ 20 
    train_limit = 80
    val_limit = 10
    test_limit = 10
    
    train_loader, val_loader = create_data_loaders(train_path, val_path, batch_size=batch_size, train_limit=train_limit, val_limit=val_limit)
    print("Data loaded")
    
    model = MLP(input_dim=6, output_dim=3).to(device)
    criterion = nn.MSELoss()
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

final_model_path = "NNAttemps/ShuffledNN2/MoreData/models/final_light_probe_model_tcnn.pth"
train_path = "D:/Projects/LightProbesWithNN/dumped_data/ShuffledData/rawraw/train"
val_path = "D:/Projects/LightProbesWithNN/dumped_data/ShuffledData/rawraw/val"
test_path = "D:/Projects/LightProbesWithNN/dumped_data/ShuffledData/rawraw/test"

if __name__ == '__main__':
    main()
