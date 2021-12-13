from agent.qpid import QPIDAgent
import numpy as np


if __name__ == '__main__':
    agent = QPIDAgent()
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    print(agent.get_actions(s, 1))
    agent.update_tables(s, 50, 0.2, 0.9)
    sl = slice(0, 9), *s
    print(agent.tables[sl])
    print(np.unravel_index(agent.tables.argmax(), agent.tables.shape))
    print(agent.tables.min())
