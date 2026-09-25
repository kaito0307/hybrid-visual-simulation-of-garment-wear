import torch
import numpy as np


class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    activation_functions = {
        "sin": torch.sin,
        "relu": torch.relu,
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "silu": torch.nn.SiLU(),
    }

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=10.0, activation="sin", fx="linear"):

        super().__init__()
        assert fx in ["linear", "finer", "sinh"]
        self.fx = fx
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        self.activation = self.activation_functions[activation]

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

            if self.fx == "finer":
                self.linear.bias.uniform_(-3.0 * np.sqrt(1.0 / self.in_features),
                                          3.0 * np.sqrt(1.0 / self.in_features))

    def forward(self, input):
        x = self.linear(input)
        if self.fx == "finer":
            x = x * (1.0 + torch.abs(x))
        elif self.fx == "sinh":
            x = torch.sinh(2.0 * x)
        return self.activation(self.omega_0 * x)


class Siren(torch.nn.Module):
    def __init__(self, in_features, coord_dim=2, hidden_features=32, hidden_layers=2, out_features=3,
                 outermost_linear=False,
                 first_omega_0=10.0, hidden_omega_0=10.0, fx="linear", activation="sin", num_frequencies=1):
        """
        fx: The preactivation function before the sin.
            "linear": The default Siren. f(x) = x https://www.vincentsitzmann.com/siren/
            "finer": f(x) = x * (1 + abs(x)) https://arxiv.org/pdf/2312.02434
            "sinh": f(x) = sinh(x) https://arxiv.org/pdf/2410.04716

        :param activation: The activation function for the MLP layers.
        :param num_frequencies: The input coordinates will be expanded using sin/cos features.
        If num_frequencies=0 only the raw coordinates will be used. The frequencies increase linearly.
        """
        super().__init__()

        self.net = []
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        coord_dim = coord_dim * max(1, 2 * num_frequencies)  # Augment coordinates with sin/cos features
        self.coord_dim = coord_dim
        self.net.append(SineLayer(in_features + coord_dim, hidden_features,
                                  is_first=True, omega_0=first_omega_0, fx=fx, activation=activation))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, fx=fx, activation=activation))

        if outermost_linear:
            final_linear = torch.nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0, fx=fx, activation=activation))

        self.net = torch.nn.Sequential(*self.net)

    def forward(self, cell_states, coords):
        """
        :param cell_states: The cell states [..., feature_dim]
        :param coords: The coordinates [..., 2] or [..., 3]
        """
        N = self.num_frequencies
        if N > 0:
            aug_coords = torch.cat([coords * 1.0 * torch.pi * i for i in range(1, N + 1)], dim=-1)
            coords = torch.cat([torch.sin(aug_coords), torch.cos(aug_coords)], dim=-1)

        x = torch.cat([coords, cell_states], dim=-1)  # Concatenate coords and cell states
        output = self.net(x)
        return output


if __name__ == "__main__":
    # Example usage
    siren = Siren(in_features=64, coord_dim=2, hidden_features=32, hidden_layers=2, out_features=4,
                  outermost_linear=True, first_omega_0=30, hidden_omega_0=30, fx="linear", activation="sin",
                  num_frequencies=2)

    cell_states = torch.randn(10, 64)  # Example cell states
    coords = (torch.rand(10, 2) - 0.5) * 2.0  # Example coordinates

    output = siren(cell_states, coords)
    print(output.shape)  # Should be [10, 4] if out_features=4
