import random

class DungeonEnv:
    def __init__(self, dungeon):
        self.dungeon = dungeon
        self.agent_pos = (1, 1)
        self.goal_pos = (len(dungeon) - 2, len(dungeon[0]) - 2)

    def reset(self):
        self.agent_pos = (1, 1)
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        new_pos = (x + (action == 1) - (action == 0), y + (action == 3) - (action == 2))

        if self.dungeon[new_pos[0]][new_pos[1]] in ['0', 'R']:  # 通路または報酬
            self.agent_pos = new_pos
            reward = 1 if self.dungeon[new_pos[0]][new_pos[1]] == 'R' else -0.1
        elif self.dungeon[new_pos[0]][new_pos[1]] == 'E':  # 敵
            self.agent_pos = new_pos
            reward = -10  # 敵に遭遇
        else:
            reward = -0.1  # 壁にぶつかる場合

        done = self.agent_pos == self.goal_pos
        return self.agent_pos, reward, done

class QLearningAgent:
    def __init__(self, actions):
        self.q_table = {}
        self.actions = actions
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(range(len(self.actions)), key=lambda a: self.q_table.get(state, [0]*len(self.actions))[a])

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, [0]*len(self.actions))
        next_q = self.q_table.get(next_state, [0]*len(self.actions))
        current_q[action] += self.alpha * (reward + self.gamma * max(next_q) - current_q[action])
        self.q_table[state] = current_q
