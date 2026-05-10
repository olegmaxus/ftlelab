from .base import *

# ============================================================
# Autoencoder
# ============================================================

class AutoEncoder(BaseNet):
    """
    Dense autoencoder.
    FTLE can be studied for:
      - encoder hidden layers
      - latent representation
      - reconstruction map
    """

    def __init__(self,
                 input_dim: int,
                 encoder_dims: Sequence[int],
                 latent_dim: int,
                 decoder_dims: Sequence[int] | None = None,
                 activation: str = "relu",
                 output_activation: str = "identity",
                 init_method: str | None = "paper",
                 negative_slope: float = 0.01,
                 dropout: float = 0.0):

        super().__init__(init_method=init_method,
                         activation_name=activation,
                         output_activation_name=output_activation,
                         negative_slope=negative_slope)

        if decoder_dims is None:
            decoder_dims = list(reversed(encoder_dims))

        self.input_dim = input_dim
        self.encoder_dims = list(encoder_dims)
        self.latent_dim = latent_dim
        self.decoder_dims = list(decoder_dims)
        self.dropout = dropout

        # encoder
        enc_layers = []
        self.encoder_activation_modules = nn.ModuleList()
        dims = [input_dim] + self.encoder_dims

        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            act = make_activation(activation, negative_slope=negative_slope)
            enc_layers.append(act)
            self.encoder_activation_modules.append(act)
            if dropout and 0.0 < dropout < 1.0:
                enc_layers.append(nn.Dropout(dropout))

        enc_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder
        dec_layers = []
        dims = [latent_dim] + self.decoder_dims

        for i in range(len(dims) - 1):
            dec_layers.append(nn.Linear(dims[i], dims[i + 1]))
            dec_layers.append(make_activation(activation, negative_slope=negative_slope))
            if dropout and 0.0 < dropout < 1.0:
                dec_layers.append(nn.Dropout(dropout))

        dec_layers.append(nn.Linear(dims[-1], input_dim))
        dec_layers.append(make_activation(output_activation, negative_slope=negative_slope))
        self.decoder = nn.Sequential(*dec_layers)

        self.apply_initialization()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def hidden_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Post-activation output of the k-th encoder hidden layer, 1-based.
        """
        if not (1 <= k <= len(self.encoder_activation_modules)):
            raise ValueError(f"k must be in [1, {len(self.encoder_activation_modules)}]")

        h = x
        act_count = 0
        for m in self.encoder:
            h = m(h)
            if m in self.encoder_activation_modules:
                act_count += 1
                if act_count == k:
                    return h
        raise RuntimeError("Could not find requested encoder hidden activation.")


# ============================================================
# Variational Autoencoder
# ============================================================

class VAE(BaseNet):
    """
    Dense VAE.
    For FTLE, use deterministic maps such as:
      - encoder trunk output
      - mu(x)
    rather than stochastic z samples.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: Sequence[int],
        latent_dim: int,
        decoder_dims: Sequence[int] | None = None,
        activation: str = "relu",
        output_activation: str = "identity",
        init_method: str | None = "paper",
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__(
            init_method=init_method,
            activation_name=activation,
            output_activation_name=output_activation,
            negative_slope=negative_slope,
        )

        if decoder_dims is None:
            decoder_dims = list(reversed(encoder_dims))

        self.input_dim = input_dim
        self.encoder_dims = list(encoder_dims)
        self.latent_dim = latent_dim
        self.decoder_dims = list(decoder_dims)
        self.dropout = dropout

        # encoder trunk
        trunk_layers = []
        self.encoder_activation_modules = nn.ModuleList()
        dims = [input_dim] + self.encoder_dims

        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            act = make_activation(activation, negative_slope=negative_slope)
            trunk_layers.append(act)
            self.encoder_activation_modules.append(act)
            if dropout and 0.0 < dropout < 1.0:
                trunk_layers.append(nn.Dropout(dropout))

        self.encoder_trunk = nn.Sequential(*trunk_layers)
        self.mu_layer = nn.Linear(dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(dims[-1], latent_dim)

        # decoder
        dec_layers = []
        dims = [latent_dim] + self.decoder_dims
        for i in range(len(dims) - 1):
            dec_layers.append(nn.Linear(dims[i], dims[i + 1]))
            dec_layers.append(make_activation(activation, negative_slope=negative_slope))
            if dropout and 0.0 < dropout < 1.0:
                dec_layers.append(nn.Dropout(dropout))

        dec_layers.append(nn.Linear(dims[-1], input_dim))
        dec_layers.append(make_activation(output_activation, negative_slope=negative_slope))
        self.decoder = nn.Sequential(*dec_layers)

        self.apply_initialization()

    def encode(self, x: torch.Tensor):
        h = self.encoder_trunk(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def latent(self, x: torch.Tensor):
        """
        Deterministic latent representation for FTLE studies.
        """
        mu, _ = self.encode(x)
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def hidden_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Post-activation output of the k-th encoder hidden layer, 1-based.
        """
        if not (1 <= k <= len(self.encoder_activation_modules)):
            raise ValueError(f"k must be in [1, {len(self.encoder_activation_modules)}]")

        h = x
        act_count = 0
        for m in self.encoder_trunk:
            h = m(h)
            if m in self.encoder_activation_modules:
                act_count += 1
                if act_count == k:
                    return h
        raise RuntimeError("Could not find requested encoder hidden activation.")


# ========================================
# Deprecated Module
# ========================================



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

            elif self.init_method in {"he", "kaiming"} or self.base_activation in {"relu", "leaky_relu"}:
                nn.init.kaiming_uniform_(module.weight, nonlinearity=self.base_activation)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

        pass

    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        y = self.forward(x)
        return torch.sign(y) if y.shape[-1] == 1 else torch.argmax(y, dim=-1)