import torch
import torch.nn as nn
import json

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)  # 新的隐藏层
        self.fc5 = nn.Linear(64, output_dim)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))  # 新的隐藏层的激活函数
        x = self.fc5(x)  # 输出层
        return x

final_model_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model.pth"
final_json_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model.json"
# Assuming the model is already trained
model = MLP(input_dim=6, output_dim=3)
model.load_state_dict(torch.load(final_model_path))

# Export weights to JSON
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().cpu().numpy().tolist()

with open(final_json_path, 'w') as f:
    json.dump(weights, f)