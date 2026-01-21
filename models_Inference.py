import torch
import torch.nn as nn
import torch.nn.init as torch_init
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Detector_VAD(nn.Module):
    def __init__(self, n_features):
        super(Detector_VAD, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)


    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

class Model_V3_Connection(nn.Module): # multiplication then Addition
    def __init__(self, n_features):
        super(Model_V3_Connection, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim=1))
        self.fc2 = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs): 
        x = self.fc1(inputs)
        att1 = self.fc_att1(inputs)
        x = (x * att1) + x  
        x = self.relu(x)
        x = self.dropout(x)
        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x  
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class violenceOneCrop(nn.Module): # NO MOVER, ESTO ES PARA MDPI y funciona para UNIFORMER-S
    def __init__(self, n_features, n_classes):
        super(violenceOneCrop, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def forward(self, x):
        # Primera capa y atención
        # x = x.unsqueeze(1)
        # print(x.size())   
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Segunda capa y atención
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Capa de salida
        x = self.fc3(x)  # Eliminado self.sigmoid
        x = self.softmax(x)  # Aplicar Softmax
        # x = x.mean(dim=1)
        return x


class violenceOneCrop_ATT(nn.Module): # multiplication then Addition
    def __init__(self, n_features, n_classes):
        super(violenceOneCrop_ATT, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim=1))
        self.fc2 = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def forward(self, inputs): 
        # Primera capa y atención
        x = self.fc1(inputs)
        att1 = self.fc_att1(inputs)
        x = (x * att1) + x  # Residual connection
        x = self.relu(x)
        x = self.dropout(x)

        # Segunda capa y atención
        # print(x.size())
        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x  # Residual connection
        x = self.relu(x)
        x = self.dropout(x)

        # Capa de salida
        x = self.fc3(x)
        x = self.softmax(x)
        # x = x.mean(dim=1)
        return x