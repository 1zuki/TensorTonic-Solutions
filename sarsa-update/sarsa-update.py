def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    q_new = [row[:] for row in q_table]

    td = reward + gamma * q_table[next_state][next_action] - q_table[state][action]

    q_new[state][action] += alpha * td

    return q_new