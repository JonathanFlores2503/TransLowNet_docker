import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class NonLocalBlock(nn.Module):
    def __init__(self, in_features):
        super(NonLocalBlock, self).__init__()
        self.in_features = in_features
        # Linear layers para generar theta, phi y g
        self.theta = nn.Linear(in_features, in_features // 2)
        self.phi = nn.Linear(in_features, in_features // 2)
        self.g = nn.Linear(in_features, in_features // 2)

        # Linear layer para restaurar las dimensiones originales
        self.W = nn.Linear(in_features // 2, in_features)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        # x: [B, N, C, D]
        batch_size, num_clips, num_crops, in_features = x.size()

        # Redimensionar para procesar el Non-Local
        x = x.view(batch_size * num_clips, num_crops, in_features)  # [B*N, C, D]

        # Proyecciones theta, phi y g
        theta_x = self.theta(x)              # [B*N, C, D//2]
        phi_x = self.phi(x).permute(0, 2, 1) # [B*N, D//2, C]
        g_x = self.g(x)                      # [B*N, C, D//2]

        # Mapa de atención
        attention = torch.bmm(theta_x, phi_x)  # [B*N, C, C]
        attention = F.softmax(attention, dim=-1)

        # Suma ponderada de g_x
        out = torch.bmm(attention, g_x)       # [B*N, C, D//2]
        out = self.W(out)                     # Restaurar a D dimensiones
        out = out + x                         # Residual connection

        # Restaurar la dimensión original
        out = out.view(batch_size, num_clips, num_crops, in_features)  # [B, N, C, D]
        return out

class Model_MultiClasses_V1(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Model_MultiClasses_V1, self).__init__()

        self.fc0 = nn.Linear(n_features, 1024)
        self.bn0 = nn.BatchNorm1d(10)  # BatchNorm para estabilizar
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(10)
        self.fc4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(32, n_classes)
        
        self.dropout = nn.Dropout(0.4)  # Dropout reducido a 40%
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        self.apply(self.weight_init)  # Inicialización de pesos

        # AdaptiveAvgPool para reducir la dimensión
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = torch.Size([batch_size, 10, 2048])
        
        # Primera capa con BatchNorm y GELU
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        # Segunda capa con BatchNorm y ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Tercera capa con BatchNorm y GELU
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        # Cuarta capa con BatchNorm y ReLU
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Quinta capa con BatchNorm y GELU
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        # Capa de salida sin Softmax (CrossEntropyLoss aplica Softmax internamente)
        x = self.fc5(x)
        
        # AdaptiveAvgPool1d para reducir la dimensión
        x = self.adaptive_pool(x.transpose(1, 2)).squeeze(-1)
        
        return x

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class Model_MultiClasses(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Model_MultiClasses, self).__init__()
        # Agregar Non-Local Block
        self.non_local = NonLocalBlock(n_features) 
        # Capas completamente conectadas
        self.fc0 = nn.Linear(n_features, 1024) # B, 10, 1024
        self.fc_att0 = nn.Sequential(nn.Linear(1024, 1024), nn.Softmax(dim=1)) # n_features


        self.fc1 = nn.Linear(1024, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(512, 512), nn.Softmax(dim=-1)) # n_features


        self.fc2 = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(32, 32), nn.Softmax(dim=-1))


        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.60)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def forward(self, x):
        # x: [B, N, C, D]
        # x = x.unsqueeze(0)
        # x = self.non_local(x)  # Salida sigue siendo [B, N, C, D]
        # x = x.squeeze(0)

        # Promediar sobre el eje de crops (dimensión 2) para obtener [B, N, D]
        # print(x.size())
        # x = x.squeeze(0)
        # print(x.size())
        # x = x.max(dim=1)[0]  # [6, 2048]
        # x = x.max(dim=0, keepdim=True)[0]  # [1, 2048]


        # Promediar sobre el eje de crops (dimensión 2) para obtener [B, N, D]
        # x = x.mean(dim=2)
        # # Promediar sobre el eje de clips (dimensión 1) para obtener [B, D]
        # x = x.mean(dim=1)

        # print(x.size())

        # Clasificación con capas FC

        x = self.fc0(x)
        att0 = self.fc_att0(x)
        x = (x * att0) + x
        x = self.relu(x)
        x = self.dropout(x)

        
        
        x = self.fc1(x)
        att1 = self.fc_att1(x)
        x = (x * att1) + x 
        x = self.relu(x)
        x = self.dropout(x)


        x = self.fc2(x)
        att2 = self.fc_att2(x)
        x = (x * att2) + x 
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.softmax(x)
        x = x.mean(dim=1)


        return x
    
class Model_V3_Classes(nn.Module): # multiplication then Addition
    def __init__(self, n_features, n_classes):
        super(Model_V3_Classes, self).__init__()

        self.fc1 = nn.Linear(n_features, 512)
        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim=1))
        self.fc2 = nn.Linear(512, 32)
        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=1))
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def forward(self, x):
        # Primera capa y atención
        inputs = x
        x = self.fc1(inputs)
        att1 = self.fc_att1(inputs)
        x = (x * att1) + x  # Residual connection
        x = self.relu(x)
        x = self.dropout(x)

        # Segunda capa y atención
        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x  # Residual connection
        
        x = self.relu(x)
        x = self.dropout(x)
        
        # Capa de salida
        x = self.fc3(x)  # Eliminado self.sigmoid
        x = self.softmax(x)  # Aplicar Softmax
        x = x.mean(dim=1)
        return x


class Model_MultiClasses_V2(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Model_MultiClasses_V2, self).__init__()

        self.fc0 = nn.Linear(n_features, 1024) # B, 10, 1024
        self.fc_att0 = nn.Sequential(nn.Linear(1024, 1024), nn.Softmax(dim=1)) # n_features

        self.fc_att1 = nn.Sequential(nn.Linear(1024, 512), nn.Softmax(dim=-1)) # n_features
        self.fc1 = nn.Linear(1024, 512)
        

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim=-1))
        self.fc2 = nn.Linear(512, 32)
        


        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.60)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.apply(weight_init)

    def forward(self, x):

        x = self.fc0(x)
        att0 = self.fc_att0(x)
        x = (x * att0) + x
        x = self.relu(x)
        x = self.dropout(x)

        att1 = self.fc_att1(x)
        x = self.fc1(x)
        x = (x * att1) + x 
        x = self.relu(x)
        x = self.dropout(x)

        att2 = self.fc_att2(x)
        x = self.fc2(x)
        x = (x * att2) + x 
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.softmax(x)
        x = x.mean(dim=1)

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
        x = x.mean(dim=1)
        return x

class ViolenceOneCropV2(nn.Module):
    def __init__(self, n_features, n_classes):
        super(ViolenceOneCropV2, self).__init__()
        self.fc1 = nn.Linear(n_features, 2048)
        self.bn1 = nn.BatchNorm1d(2048)  
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024) 
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, n_classes)
        
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        self.apply(weight_init)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.softmax(x)
        return x


class ImprovedViolenceOneCrop(nn.Module):
    def __init__(self, n_features, n_classes):
        super(ImprovedViolenceOneCrop, self).__init__()
        # Arquitectura ajustada
        self.fc1 = nn.Linear(n_features, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, n_classes)

        # Regularización y activaciones
        self.dropout = nn.Dropout(0.5)  # Incrementado para capas finales
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=-1)

        # Inicialización de pesos
        self.apply(weight_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.softmax(x)
        return x
    
class Model_V2(nn.Module): # multiplication then Addition
    def __init__(self, n_features):
        super(Model_V2, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)

        self.fc_att1 = nn.Sequential(nn.Linear(n_features, 512), nn.Softmax(dim = 1))

        self.fc2 = nn.Linear(512, 32)

        self.fc_att2 = nn.Sequential(nn.Linear(512, 32), nn.Softmax(dim = 1))

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

        x = x.mean(dim = 1)

        return x
