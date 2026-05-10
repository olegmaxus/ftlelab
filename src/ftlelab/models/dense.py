from .base import *

# ============================================================
# Dense network with arbitrary widths
# ============================================================

class DenseNet(BaseNet):
    """
    Fully-connected network with arbitrary layer widths.
    Example:
        DenseNet(layer_dims=[2, 32, 64, 32, 1], activation="tanh", output_activation="tanh")
    """

    def __init__(self,
                 layer_dims: Sequence[int],
                 activation: str = "tanh",
                 output_activation: str = "identity",
                 dropout: float = 0.0,
                 init_method: str | None = "paper",
                 negative_slope: float = 0.01):

        super().__init__(init_method=init_method,
                         activation_name=activation,
                         output_activation_name=output_activation,
                         negative_slope=negative_slope)

        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least input and output dimensions.")

        self.layer_dims = list(layer_dims)
        self.input_dim = self.layer_dims[0]
        self.output_dim = self.layer_dims[-1]
        self.hidden_dims = self.layer_dims[1:-1]
        self.hidden_depth = len(self.hidden_dims)
        self.dropout = dropout

        layers = []
        self.hidden_activation_modules = nn.ModuleList()

        for i in range(len(self.layer_dims) - 2):
            layers.append(nn.Linear(self.layer_dims[i], 
                                    self.layer_dims[i + 1]))
            act = make_activation(activation, 
                                  negative_slope=negative_slope)
            layers.append(act)

            self.hidden_activation_modules.append(act)

            if dropout and 0.0 < dropout < 1.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(self.hidden_dims[-1], 
                                self.output_dim))
        layers.append(make_activation(output_activation, 
                                      negative_slope=negative_slope))

        self.net = nn.Sequential(*layers)
        self.apply_initialization()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def hidden_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Post-activation output of the k-th hidden layer, 1-based.
        """
        if not (1 <= k <= self.hidden_depth):
            raise ValueError(f"k must be in [1, {self.hidden_depth}]")

        h = x
        act_count = 0
        for m in self.net:
            h = m(h)
            if m in self.hidden_activation_modules:
                act_count += 1
                if act_count == k:
                    return h

        raise RuntimeError("Could not find requested hidden activation.")



    