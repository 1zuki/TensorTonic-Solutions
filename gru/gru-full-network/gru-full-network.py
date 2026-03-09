import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last)."""
        # YOUR CODE HERE
        batches, time, _ = X.shape
        h = np.zeros((batches, self.hidden_dim))

        hidden = []

        for t in range(time):
            x_t = X[:, t, :]
            concat = np.concatenate((h, x_t), axis = 1)

            r_t = sigmoid(concat @ self.W_r.T + self.b_r)
            z_t = sigmoid(concat @ self.W_z.T + self.b_z)

            concat = np.concatenate((r_t * h, x_t), axis = 1)

            h_tilde = np.tanh(concat @ self.W_h.T + self.b_h)
            h = z_t * h + (1 - z_t) * h_tilde

            hidden.append(h)

        h_all = np.stack(hidden, axis = 1)
        h_last = h

        y = h_all @ self.W_y.T + self.b_y

        return y, h_last

            