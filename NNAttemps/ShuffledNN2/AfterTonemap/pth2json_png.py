import torch
import json

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def save_model_to_json(pth_file, json_file):
    model = MLP(6, 3)
    model.load_state_dict(torch.load(pth_file))
    model.eval()
    
    weights = {
        'fc1': model.fc1.weight.detach().numpy().tolist(),
        'fc1_bias': model.fc1.bias.detach().numpy().tolist(),
        'fc2': model.fc2.weight.detach().numpy().tolist(),
        'fc2_bias': model.fc2.bias.detach().numpy().tolist(),
        'fc3': model.fc3.weight.detach().numpy().tolist(),
        'fc3_bias': model.fc3.bias.detach().numpy().tolist(),
        'fc4': model.fc4.weight.detach().numpy().tolist(),
        'fc4_bias': model.fc4.bias.detach().numpy().tolist(),
    }
    
    with open(json_file, 'w') as f:
        json.dump(weights, f)


model_file_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model_morePNG.pth"
model_json_path = "NNAttemps/ShuffledNN2/models/final_light_probe_model_morePNG.json"
if __name__ == "__main__":
    save_model_to_json("model.pth", "model.json")
