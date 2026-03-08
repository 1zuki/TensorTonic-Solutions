import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE
        batch, time, dim = X.shape
        hidden = []

        if h_0 is None:
            h = np.zeros((batch, self.hidden_dim))
        else:
            h = h_0

        for t in range(time):
            x_t = X[:, t, :]
            
            h = np.tanh(x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            
            hidden.append(h)
    
        h_all = np.stack(hidden, axis=1)
        h_final = h

        y_all = h_all @ self.W_hy.T + self.b_y

        return y_all, h_final