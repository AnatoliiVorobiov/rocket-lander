from agent.qpid import QPIDAgent


if __name__ == '__main__':
    agent = QPIDAgent()
    print(agent.tables.shape)
    s = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    print(agent.tables[s])
