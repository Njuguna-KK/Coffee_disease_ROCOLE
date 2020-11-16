import torch.nn as nn


class Net(nn.Module):
    def __init__(self, params):
        """
        We define a fully connected network that predicts the category of disease of coffee leaves:
            LINEAR -> RELU  -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_classes = params.num_classes
        self.img_dimension = params.img_dimension
        self.first_hidden_size = params.first_hidden_size
        self.second_hidden_size = params.second_hidden_size
        self.third_hidden_size = params.third_hidden_size
        self.depth = params.depth
        self.batch_size = params.batch_size

        self.linear1 = nn.Linear(self.img_dimension * self.img_dimension * self.depth, self.first_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.first_hidden_size, self.second_hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(self.second_hidden_size, self.third_hidden_size)
        self.relu = nn.ReLU()
        self.linear4 = nn.Linear(self.third_hidden_size, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        num_examples = x.shape[0]
        out = x.view(num_examples, -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.softmax(out)
        return out
