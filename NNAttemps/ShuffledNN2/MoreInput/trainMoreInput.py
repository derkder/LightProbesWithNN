import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt

def get_exr_file_paths(batch_folder_path, keywords):
    file_paths = {}
    for file_name in os.listdir(batch_folder_path):
        for keyword in keywords:
            if keyword in file_name and file_name.endswith('.exr'):
                file_paths[keyword] = os.path.join(batch_folder_path, file_name)
    return file_paths

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
        self.folder_path = folder_path
        self.data_files = []
        self.keywords = ["probePoses", "rayDirs", "output", "diffuse", "roughnessemmisive", "specular"]

        for batch_folder in os.listdir(folder_path):
            batch_folder_path = os.path.join(folder_path, batch_folder)
            exr_paths = get_exr_file_paths(batch_folder_path, self.keywords)
            if all(keyword in exr_paths for keyword in self.keywords):
                self.data_files.append(exr_paths)
                if limit and len(self.data_files) >= limit:
                    break

        print(f"Found {len(self.data_files)} data file sets in {folder_path}.")

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        exr_paths = self.data_files[idx]
        
        images = {}
        for key, path in exr_paths.items():
            _, image = read_exr(path)
            images[key] = image.reshape(-1, 3)

        images["diffuse"] *= 0.3
        # images["diffuse"] *= 0.01
        # images["roughnessemmisive"] *= 0.1
        # images["specular"] *= 0.1
        input_data = np.concatenate(
            [images["probePoses"], images["rayDirs"], images["diffuse"], images["roughnessemmisive"], images["specular"]], 
            axis=1
        ).reshape(-1, 15)
        color_data = images["output"].reshape(-1, 3)
        
        return torch.tensor(input_data, dtype=torch.float16), torch.tensor(color_data, dtype=torch.float16)

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

def train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps=4):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")
    
    for batch_idx, (input_data, target) in enumerate(train_loader):
        input_data, target = input_data.to(device), target.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 自动混合精度上下文管理器
            outputs = model(input_data)
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        torch.cuda.empty_cache()  # 尝试释放未使用的显存

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
            torch.cuda.empty_cache()  # 尝试释放未使用的显存
   
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
            torch.cuda.empty_cache()  # 尝试释放未使用的显存
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
    
    batch_size = 2
    train_limit = 400
    val_limit = 50
    test_limit = 50
    
    train_loader, val_loader, test_loader = create_data_loaders(train_path, val_path, test_path, batch_size=batch_size, train_limit=train_limit, val_limit=val_limit, test_limit=test_limit)
    print("Data loaded")
    
    model = MLP(input_dim=15, hidden_dim=128, output_dim=3).to(device)
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
    test_loss = test(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

final_model_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model_moreinput2.pth"
train_path = "C:/Files/CGProject/NNLightProbes/dumped_data/ShuffledData/raw/train"
val_path = "C:/Files/CGProject/NNLightProbes/dumped_data/ShuffledData/raw/val"
test_path = "C:/Files/CGProject/NNLightProbes/dumped_data/ShuffledData/raw/test"

if __name__ == '__main__':
    main()
