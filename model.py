import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cov_net = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, stride=1),
            nn.ReLU(),
            nn.Dropout(.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*2*2, 300),
            nn.ReLU(),
            nn.Linear(300, 4)
        )

    def forward(self, x):
        x = self.cov_net(x)
        x = x.view(-1, 64*2*2)
        x = self.fc(x)
        return x

    def predict(self, x):
        scores = self.forward(x)
        _, predictions = scores.max(1)
        return predictions

    def predict_proba(self, x):
        """
            Ouput the probability assigned to the predicted class label
        """
        scores = self.forward(x)
        scores = F.softmax(scores, dim=1)
        probs, predictions = scores.max(1)
        return probs, predictions
