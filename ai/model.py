import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

names_df = pd.read_csv('names.csv')

class Net(nn.Module):

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    nr_of_classes = names_df['name'].count() 
    classes = {
        "0":"cat",
        "1":"dog",
    }
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(512, self.nr_of_classes) #class nr
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x