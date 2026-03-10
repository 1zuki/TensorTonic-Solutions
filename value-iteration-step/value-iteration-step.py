def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    new_values = []

    for s in range(len(transitions)):
        best = float("-inf")

        for a in range(len(transitions[s])):
            
            expected = 0

            for s_next in range(len(transitions[s][a])):
                expected += transitions[s][a][s_next] * values[s_next]

            q = rewards[s][a] + gamma * expected

            best = max(best, q)

        new_values.append(best)

    return new_values