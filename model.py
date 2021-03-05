import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Layer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, activation, dropout, pool=None):
        super(Conv_Layer, self).__init__()

        # Use padding so that feature maps don't decrease in size throughout network. Makes things wrt dimensions nice when moving to FC layers.
        # Math says that we should pad by (kernel_size[0]//2, kernel_size[1]//2) to keep dimensions the same, in our case this is (1, 2)  
        # However, paper says they only pad along time axis, so number of features will decrease 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=(0,kernel[1]//2))

        self.maxout = activation == "maxout"
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            # Need to use module instead of functional since it has a parameter and needs to be pushed to cuda
            self.activation = nn.PReLU(init=0.1)
        elif activation == "maxout":
            # Better results (theoretically), worse memory usage
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=(0,kernel[1]//2))
        else:
            raise Exception("Not a supported activation function. Choose between 'relu', 'prelu', or 'maxout'.")
       
        # Using Dropout module vs functional will automatically disable it during model.eval()
        self.dropout = nn.Dropout(p=dropout)

        # Pool reduces dimension: ((old - (pool_dim - 1) - 1) / pool_dim) + 1, turns out to do 118 -> 39 in our case
        self.pool = pool

    def forward(self, x):
        if self.maxout:
            x = self.conv_maxout(x)
        else:
            x = self.conv(x)
            x = self.activation(x)

        if self.pool:
            x = nn.MaxPool2d(kernel_size=self.pool)(x)

        x = self.dropout(x)
        return x

    def conv_maxout(self, x):
        # 2-way maxout creates 2 versions of a layer and maxes them, that's it
        
        x1, x2 = self.conv(x), self.conv2(x)
        return torch.maximum(x1, x2)

class FC_Layer(nn.Module):

    def __init__(self, in_dim, out_dim, activation, dropout):
        super(FC_Layer, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        
        self.maxout = activation == "maxout"
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            # Need to use module instead of functional since it has a parameter and needs to be pushed to cuda
            self.activation = nn.PReLU(init=0.1)
        elif activation == "maxout":
            # Better results (theoretically), worse memory usage
            self.fc2 = nn.Linear(in_dim, out_dim)
        else:
            raise Exception("Not a supported activation function. Choose between 'relu', 'prelu', or 'maxout'.")
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.maxout:
            x = self.fc_maxout(x)
        else:
            x = self.fc(x)
            x = self.activation(x)
        x = self.dropout(x)
        return x

    def fc_maxout(self, x):
        
        x1, x2 = self.fc(x), self.fc2(x)
        return torch.maximum(x1, x2)

class ASR_1(nn.Module):

        """
        Unlike the other layers, the first convolutional layer is followed by a pooling layer. 
        The pooling size is 3 × 1, which means we only pool over the frequency axis. The filter size is 3 × 5 across the layers. 
        The model has 256 feature maps in the first four convolutional layers and 256 feature maps in the remaining six convolutional layers. 
        Each fully-connected layer has 1024 units. Maxout with 2 piece-wise linear functions is used as the activation function.

        We differ from the given model by using ReLU instead of Maxout for the activation function. This is simply due to memory constraints on 
        available hardware. 
        """ 

        def __init__(self, in_dim, num_classes, num_features, activation, dropout):
            
            super(ASR_1, self).__init__()

            self.cnn_layers = nn.Sequential(
                            Conv_Layer(in_channels=in_dim, out_channels=128, kernel=(3,5), activation="relu", dropout=0, pool=(3,1)),
                            Conv_Layer(128, 128, (3,5), activation, dropout),
                            Conv_Layer(128, 128, (3,5), activation, dropout),
                            Conv_Layer(128, 128, (3,5), activation, dropout),
                            Conv_Layer(128, 256, (3,5), activation, dropout),
                            Conv_Layer(256, 256, (3,5), activation, dropout),
                            Conv_Layer(256, 256, (3,5), activation, dropout),
                            Conv_Layer(256, 256, (3,5), activation, dropout),
                            Conv_Layer(256, 256, (3,5), activation, dropout),
                            Conv_Layer(256, 256, (3,5), activation, dropout)
            )

            # For feature maps, Conv2d: 3x5 conv (with appropriate padding) will decrease frequency dim by 2
            #                   MaxPool2d: 3x1 pooling decreases frequency dim by 2
            # Features dimenion of resulting feature map: num_features = 120 -> 39 (after first conv + pool) - 2 (conv[2-10]) * 9 = 39 - 18 = 21
            # So in our use case, first fc layer should have dimensions = 256 * 21 = 5376

            # TODO: automate this math instead of hard coding the value. would require passing in number of certain kinds of layers and their underlying details (not super important)

            self.fc_layers = nn.Sequential(
                            FC_Layer(in_dim=256 * 21, out_dim=1024, activation=activation, dropout=dropout),
                            FC_Layer(1024, 1024, activation, dropout),
                            FC_Layer(1024, 1024, activation, dropout),
                            FC_Layer(1024, num_classes, activation, 0)
            )

        def forward(self, x):
            x = self.cnn_layers(x) # output shape (batch_size, channels, features, time)
            x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) # flattens channel and feature dimensions into one
            x = x.transpose(1, 2) # (batch_size, time, flattened_features)

            x = self.fc_layers(x) # FC layer input: (N, *, H_{in}) where * means any number of additional dimensions and H_in=in_features

            x = F.log_softmax(x, dim=2)
            return x