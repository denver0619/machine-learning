import torch.nn as nn

# Score 0.77033
# epoch 10000
# pytorch BCEloss SGD lr=0.0001 output-threshold=0.5
class SurvivedModel1(nn.Module):
    def __init__(self, MAXLENGTH=1000):
        super(SurvivedModel1, self).__init__()
        self.layer1 = nn.Linear(MAXLENGTH, 48)
        self.layer2 = nn.Linear(48, 1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    
class SurvivedModel(nn.Module):
    def __init__(self, MAXLENGTH=1000):
        super(SurvivedModel, self).__init__()
        self.layer1 = nn.Linear(MAXLENGTH, 24)
        self.layer2 = nn.Linear(24, 12)
        self.layer3 = nn.Linear(12, 4)
        self.layer4 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        return x