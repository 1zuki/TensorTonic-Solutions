import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    V = np.array(V)
    states = np.array(states)

    A = []

    for t in range(len(states)):
        Gt = 0
        power = 0

        for k in range(t, len(rewards)):
            Gt += (gamma ** power) * rewards[k]
            power += 1

        A.append(Gt - V[states[t]])

    return np.array(A)