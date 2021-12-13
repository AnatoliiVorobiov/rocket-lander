from agent.qpid import QPIDAgent


if __name__ == '__main__':
    agent = QPIDAgent()
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    print(agent.get_coefficients(s, 0.5))
    sl = (slice(0, 9), 0, 0, 0, 0, 0, 0, 0, 0)
    print(agent.tables[sl])
