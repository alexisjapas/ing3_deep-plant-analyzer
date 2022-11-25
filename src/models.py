from torch import nn


class FlatCNN(nn.Module):
    def __init__(self, in_size, in_channels, hidden_channels, n_layers, n_outputs):
        super().__init__()

        # Relu
        self.relu = nn.ReLU(inplace=True)

        # Input layer
        self.input_layer = nn.Conv2d(in_channels=in_channels,
                                     out_channels=hidden_channels,
                                     kernel_size=3,
                                     padding='same')

        # Hidden layers
        conv2d_relu = []
        for i in range(n_layers):
            conv2d_relu.append(nn.Conv2d(in_channels=hidden_channels,
                                         out_channels=hidden_channels,
                                         kernel_size=3,
                                         padding='same'))
            conv2d_relu.append(nn.ReLU())
        self.conv2d_relu = nn.Sequential(*conv2d_relu)

        # Output layer
        self.output_layer = nn.Linear(in_size*in_size*hidden_channels, n_outputs)

    def forward(self, x):
        noisy_input = x
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.conv2d_relu(x)
        return self.output_layer(x)
