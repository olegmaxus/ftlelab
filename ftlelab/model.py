import torch
import torch.nn as nn

ACTS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus
}

class CustomDNN(nn.Module):
    def __init__(self, hidden_dim, hidden_depth, input_dim, output_dim, base_activation="tanh", last_activation="tanh", init_method="paper", dropout=0.0):
        """
        hidden_dim: the hidden dimension of the network (the number of neurons in each layer, N)
        depth: the network's hidden_depth (the number of hidden layers, L)
        """
        super(CustomDNN, self).__init__()
        self.layers = [] # nn.ModuleList()
        activation = base_activation.lower()

        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = dropout
        # Set default activations to tanh
        
        self.base_activation = ACTS.get(activation, nn.Tanh)

        # self.activations = [self.base_activation()] * hidden_depth + [nn.Tanh()]
        self.init_method = init_method.lower() if init_method else None

        last = self.input_dim
        for _ in range(self.hidden_depth):
            self.layers.append(nn.Linear(last, self.hidden_dim))
            self.layers.append(self.base_activation())
            if self.dropout and self.dropout > 0.0 and self.dropout < 1.0:
                self.layers.append(nn.Dropout(p=self.dropout))
            last = self.hidden_dim

        # Output layer
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers.append(nn.Tanh())

        self.net = nn.Sequential(*self.layers)

        # Apply paper-based initialization
        if self.init_method: 
            self.apply(self._initialize_weights)
        else:
            pass

    # def hidden_last(self, x):
    #     # returns post-activation of last hidden layer (dropout disabled in eval: use model.eval())
    #     h = x
    #     act_count = 0
    #     for m in self.net:
    #         h = m(h)
    #         if isinstance(m, tuple(ACTS.values())):
    #             act_count += 1
    #             if act_count == self.hidden_depth:  # number of hidden layers
    #                 return h
    #     raise RuntimeError("Could not find last hidden activation.")
    
    def hidden_k(self, x, k: int):
        """
        Return post-activation tensor of the k-th hidden layer (1-based).
        Assumes hidden blocks are Linear -> Activation -> (optional Dropout),
        followed by the output Linear and a final activation.
        """
        if not (1 <= k <= self.hidden_depth):
            raise ValueError(f"k must be in [1, {self.hidden_depth}]")

        h = x
        act_count = 0
        for m in self.net:
            h = m(h)
            if isinstance(m, tuple(ACTS.values())):
                act_count += 1
                if act_count == k:
                    return h
        raise RuntimeError("Could not find the requested hidden activation.")

    def _initialize_weights(self, module):
        """
        Custom weight initialization as per the paper.
        Weights: Gaussian distribution with variance 1/N
        Biases: Zero initialization
        """

        ### init.method_ -> modifies inplace

        if isinstance(module, nn.Linear):
            if self.init_method == "paper":
                fan_in = module.weight.size(1)  # Number of input features
                std = (1.0 / fan_in) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)  # Gaussian with variance 1/N
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # Biases initialized to zero

            elif self.init_method in {"glorot", "xavier"}: # for tanh
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            elif self.init_method in {"he", "kaiming"} or self.activation in {"relu", "leaky_relu"}:
                nn.init.kaiming_uniform_(module.weight, nonlinearity=self.activation)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

        pass

    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        y = self.forward(x)
        return torch.sign(y) if y.shape[-1] == 1 else torch.argmax(y, dim=-1)
    
