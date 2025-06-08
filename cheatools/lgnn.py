import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Linear
from tqdm import tqdm

class lGNN(torch.nn.Module):
    def __init__(self, arch=None, trained_state=None):
        super(lGNN, self).__init__()
        device = torch.device('cpu')
        if arch == None and trained_state != None:
            arch = trained_state['arch']
            del trained_state['arch']
 
        self.input_dim = arch.get('input_dim', 6)  
        self.out_dim = 1
        self.act = arch['act']
        self.att = AttentionalAggregation(Linear(arch['conv_dim'], 1))

        # Set up gated graph convolution with input dimension
        self.input_transform = Linear(self.input_dim, arch['conv_dim'])
        self.conv = GatedGraphConv(out_channels=arch['conv_dim'], 
                                 aggr='add', 
                                 num_layers=arch['n_conv_layers'])

        # Set up hidden layers
        self.fc_list = torch.nn.ModuleList()
        if arch['n_hidden_layers'] > 0:
            for i in range(arch['n_hidden_layers']):
                self.fc_list.append(Linear(arch['conv_dim'], arch['conv_dim']))
    
        # Final output layer
        self.lin_out = Linear(arch['conv_dim'], self.out_dim)
        
        device = torch.device('cpu')
        self.to(device)

        if trained_state != None:
            self.load_state_dict(trained_state)

    def forward(self, data):
        """
        Forward pass of the model.
        """
        # Ensure input features are float tensors
        if not isinstance(data.x, torch.Tensor):
            data.x = torch.tensor(data.x, dtype=torch.float32)
        
        # Transform input features to conv_dim
        out = self.input_transform(data.x)
        
        # Gated Graph Convolutions
        out = self.conv(out, data.edge_index)

        # Global attention pooling
        out = self.att(out, data.batch)

        # Hidden layers
        for layer in self.fc_list:
            out = layer(out)
            out = getattr(torch.nn.functional, self.act)(out)

        # Output layer
        out = self.lin_out(out)

        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

    def train4epoch(self, loader, batch_size, optimizer):
        """
        Train the model for a single epoch.
        """
        self.train()
        pred, target = [], []

        for data in loader:
            optimizer.zero_grad()

            # predict
            predicted = self(data)

            # compute mean squared error
            loss = torch.nn.MSELoss()(predicted.reshape([-1]), data.y)

            # update step
            loss.backward()
            optimizer.step()

            pred += predicted.reshape([-1]).tolist()
            target += data.y.tolist()

        # return mean absolute error
        L1Loss = np.mean(abs(np.array(pred) - np.array(target)))
        return L1Loss

    def test(self, loader, batch_size):
        """
        Predict on provided dataloader.
        """
        self.eval()
        pred, target, ads = [], [], []

        for data in loader:
            pred += self(data).reshape([-1]).tolist()
            target += data.y.tolist()
            try:
                ads += data.ads
            except AttributeError:
                ads += ['N/A']

        return pred, target, ads

    def predict(self, graphs, tqdm_bool=True):
        """
        Predict on provided list of graphs.
        """
        self.eval()
        loader = DataLoader(graphs, batch_size=256)
        pred = []
        for data in tqdm(loader, total=len(loader), disable=not tqdm_bool):
            pred += self(data).reshape([len(data)]).tolist()
        return np.array(pred)