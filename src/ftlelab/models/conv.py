from .base import *

# ============================================================
# Convolutional network without skip-connections
# ============================================================

class ConvNet(BaseNet):
    """
    Plain CNN + MLP head, no skip-connections.

    conv_specs: list of dicts, e.g.
        [
            {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "pool": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool": 2},
        ]

    input_shape: (C, H, W)
    head_dims: e.g. [128, 10]
    """

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 conv_specs: Sequence[dict],
                 head_dims: Sequence[int],
                 activation: str = "relu",
                 output_activation: str = "identity",
                 init_method: str | None = "he",
                 negative_slope: float = 0.01,
                 dropout: float = 0.0):

        super().__init__(init_method=init_method,
                         activation_name=activation,
                         output_activation_name=output_activation,
                         negative_slope=negative_slope)

        self.input_shape = input_shape
        self.conv_specs = list(conv_specs)
        self.head_dims = list(head_dims)
        self.dropout = dropout

        c_in = input_shape[0]
        self.blocks = nn.ModuleList()

        for spec in self.conv_specs:
            c_out = spec["out_channels"]
            k = spec.get("kernel_size", 3)
            s = spec.get("stride", 1)
            p = spec.get("padding", k // 2)
            pool = spec.get("pool", None)
            use_bn = spec.get("batchnorm", False)

            block_layers = [nn.Conv2d(c_in,
                                      c_out,
                                      kernel_size=k,
                                      stride=s,
                                      padding=p)]
            if use_bn:
                block_layers.append(nn.BatchNorm2d(c_out))

            block_layers.append(make_activation(activation, negative_slope=negative_slope))
            
            if pool is not None:
                block_layers.append(nn.MaxPool2d(pool))
            
            if dropout and 0.0 < dropout < 1.0:
                block_layers.append(nn.Dropout2d(dropout))

            self.blocks.append(nn.Sequential(*block_layers))
            c_in = c_out

        self.flatten = nn.Flatten()

        # infer flattened dimension
        with torch.no_grad():
            _ = torch.zeros(1, *input_shape)
            h = _
            for block in self.blocks:
                h = block(h)
            flat_dim = h.reshape(1, -1).shape[1]

        dense_layers = []
        dims = [flat_dim] + self.head_dims
        self.head_activation_modules = nn.ModuleList()

        for i in range(len(dims) - 2):
            dense_layers.append(nn.Linear(dims[i], dims[i + 1]))
            act = make_activation(activation, negative_slope=negative_slope)
            dense_layers.append(act)
            self.head_activation_modules.append(act)

            if dropout and 0.0 < dropout < 1.0:
                dense_layers.append(nn.Dropout(dropout))

        dense_layers.append(nn.Linear(dims[-2], dims[-1]))
        dense_layers.append(make_activation(output_activation, negative_slope=negative_slope))

        self.head = nn.Sequential(*dense_layers)
        self.apply_initialization()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.flatten(h)
        return self.head(h)

    def hidden_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Output of the k-th conv block, 1-based.
        """
        if not (1 <= k <= len(self.blocks)):
            raise ValueError(f"k must be in [1, {len(self.blocks)}]")

        h = x
        for i, block in enumerate(self.blocks, start=1):
            h = block(h)
            if i == k:
                return h
        raise RuntimeError("Could not find requested conv block.")


