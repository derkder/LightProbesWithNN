import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import OpenEXR
import Imath
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        self.colors = []
        self.load_data(folder_path)
    
    def load_data(self, folder_path):
        batch_folders = os.listdir(folder_path)
        for batch_folder in batch_folders:
            batch_folder_path = os.path.join(folder_path, batch_folder)
            num_data_points = 0
            hitpoints = []
            raydirs = []
            with open(os.path.join(batch_folder_path, "data.json"), 'r') as f:
                json_data = json.load(f)
                num_data_points = json_data["num"]
                for i in range(num_data_points):
                    hitpoint = np.array(json_data["hitpoints"][i])
                    raydir = np.array(json_data["raydir"][i])
                    hitpoints.append(hitpoint)
                    raydirs.append(raydir)
                    # self.data.append((hitpoint, raydir))

            color_exr_path = os.path.join(batch_folder_path, "color.exr")
            diffuse_exr_path = os.path.join(batch_folder_path, "diffuse.exr")
            roughEmmi_exr_path = os.path.join(batch_folder_path, "roughEmmi.exr")
            specular_exr_path = os.path.join(batch_folder_path, "specular.exr")

            _, color_image = read_exr(color_exr_path)
            flat_colors = color_image.reshape(-1, 3)
            self.colors.extend(flat_colors[:num_data_points])
            _, diffuse_image = read_exr(diffuse_exr_path)
            flat_diffuses = diffuse_image.reshape(-1, 3)
            _, roughEmmi_image = read_exr(roughEmmi_exr_path)
            flat_roughEmmis = roughEmmi_image.reshape(-1, 3)
            _, specular_image = read_exr(specular_exr_path)
            flat_speculars = specular_image.reshape(-1, 3)
            self.data.extend(zip(hitpoints[:num_data_points], raydirs[:num_data_points], flat_diffuses[:num_data_points], flat_roughEmmis[:num_data_points], flat_speculars[:num_data_points]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hitpoint, raydir, diffuse, roughEmmi, specular = self.data[idx]
        color = self.colors[idx]
        return torch.tensor(hitpoint, dtype=torch.float32), torch.tensor(raydir, dtype=torch.float32), torch.tensor(diffuse, dtype=torch.float32), torch.tensor(roughEmmi, dtype=torch.float32), torch.tensor(specular, dtype=torch.float32), torch.tensor(color, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train(model, device, train_loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (hitpoint, raydir, diffuse, roughEmmi, specular, target) in enumerate(train_loader):
        hitpoint, raydir, diffuse, roughEmmi, specular, target = hitpoint.to(device), raydir.to(device), diffuse.to(device), roughEmmi.to(device), specular.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(torch.cat((hitpoint, raydir, diffuse, roughEmmi, specular), dim=1))
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    average_loss = running_loss / len(train_loader)
    return average_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for hitpoint, raydir, diffuse, roughEmmi, specular, target in val_loader:
            hitpoint, raydir, diffuse, roughEmmi, specular, target = hitpoint.to(device), raydir.to(device), diffuse.to(device), roughEmmi.to(device), specular.to(device), target.to(device)
            output = model(torch.cat((hitpoint, raydir, diffuse, roughEmmi, specular), dim=1))
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader)
    return val_loss

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for hitpoint, raydir, diffuse, roughEmmi, specular, target in test_loader:
            hitpoint, raydir, diffuse, roughEmmi, specular, target = hitpoint.to(device), raydir.to(device), diffuse.to(device), roughEmmi.to(device), specular.to(device), target.to(device)
            output = model(torch.cat((hitpoint, raydir, diffuse, roughEmmi, specular), dim=1))
            test_loss += criterion(output, target).item()
    test_loss /= len(test_loader)
    return test_loss

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    train_dataset = CustomDataset(os.path.join(output_path, "train"))
    val_dataset = CustomDataset(os.path.join(output_path, "val"))
    test_dataset = CustomDataset(os.path.join(output_path, "test"))
    print("dataloaded")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model = MLP(input_dim=15, hidden_dim=256, output_dim=3).to(device)  # input_dim 修改为 15
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, criterion, optimizer, scaler, epoch)
        val_loss = validate(model, device, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), final_model_path)
    
    print("Training complete. Testing on test dataset...")
    test_loss = test(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Plot validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

final_model_path = "NNAttemps/ShuffledNNMoreInput/models/final_light_probe_model_6.pth"
# path to processed data
output_path = "dumped_data/tempFullData/processed_real/"  
if __name__ == '__main__':
    main()
