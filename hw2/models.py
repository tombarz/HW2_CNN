import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        blocks.append(Linear(in_features, hidden_features[0]))
        for i in range(len(hidden_features) - 1):
            if activation == 'relu':
                blocks.append(ReLU())
            else:
                blocks.append(Sigmoid())
            blocks.append(Dropout(dropout))
            blocks.append(Linear(hidden_features[i], hidden_features[i + 1]))
        # ========================
        if activation == 'relu':
            blocks.append(ReLU())
        else:
            blocks.append(Sigmoid())
        blocks.append(Dropout(dropout))
        blocks.append(Linear(hidden_features[-1], num_classes))
        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        for i in range(len(self.filters)):
            layers.append(nn.Conv2d(in_channels, self.filters[i], 3, padding=1)) #preserve dimensions with padding
            layers.append(nn.ReLU())
            in_channels = self.filters[i]
            if (i + 1) % self.pool_every == 0: #pool every P layers
                layers.append(nn.MaxPool2d(2))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======

        #calculate number of features:
        num_pools = len(self.filters) // self.pool_every
        h_last_conv = in_h // (2 ** num_pools)
        w_last_conv = in_w // (2 ** num_pools)
        num_features = self.filters[-1] * h_last_conv * w_last_conv

        #add layers:
        layers.append(nn.Linear(num_features, self.hidden_dims[0]))
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes)) #last layer

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======

        conv_output = self.feature_extractor(x)
        #TODO remove print
        #print(conv_output.shape)
        #TODO remove print
        conv_output = conv_output.view(conv_output.size(0), -1) #flatten
        out = self.classifier(conv_output)



        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
   # raise NotImplementedError()
    # ========================

