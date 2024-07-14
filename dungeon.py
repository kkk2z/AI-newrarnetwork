import random

def create_dungeon(width, height, layers):
    dungeon = []

    for layer in range(layers):
        layer_map = [['1' for _ in range(width)] for _ in range(height)]

        # 通路を生成
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                layer_map[i][j] = '0' if random.random() > 0.3 else '1'

        layer_map[1][1] = '0'  # スタート地点
        layer_map[height - 2][width - 2] = '0'  # ゴール地点

        # 報酬と敵を配置
        for _ in range(5):  # 報酬の数
            x, y = random.randint(1, height - 2), random.randint(1, width - 2)
            while layer_map[x][y] != '0':
                x, y = random.randint(1, height - 2), random.randint(1, width - 2)
            layer_map[x][y] = 'R'  # 報酬を配置

        for _ in range(3):  # 敵の数
            x, y = random.randint(1, height - 2), random.randint(1, width - 2)
            while layer_map[x][y] != '0':
                x, y = random.randint(1, height - 2), random.randint(1, width - 2)
            layer_map[x][y] = 'E'  # 敵を配置

        dungeon.append(layer_map)

    return dungeon

class DungeonEnv:
    def __init__(self, dungeon):
        self.dungeon = dungeon
        self.agent_pos = (1, 1, 0)  # (x, y, layer)
        self.goal_pos = (len(dungeon[0]) - 2, len(dungeon[0][0]) - 2, 0)

    def reset(self):
        self.agent_pos = (1, 1, 0)
        return self.agent_pos

    def step(self, action):
        x, y, layer = self.agent_pos
        new_pos = (x + (action == 1) - (action == 0),
                    y + (action == 3) - (action == 2),
                    layer)

        # 階層の上下移動
        if action == 4 and layer < len(self.dungeon) - 1:  # 下の階層へ
            new_pos = (x, y, layer + 1)
        elif action == 5 and layer > 0:  # 上の階層へ
            new_pos = (x, y, layer - 1)

        if (0 <= new_pos[0] < len(self.dungeon[0]) and
            0 <= new_pos[1] < len(self.dungeon[0][0]) and
            0 <= new_pos[2] < len(self.dungeon)):
            if self.dungeon[new_pos[2]][new_pos[0]][new_pos[1]] in ['0', 'R']:
                self.agent_pos = new_pos
                reward = 1 if self.dungeon[new_pos[2]][new_pos[0]][new_pos[1]] == 'R' else -0.1
            elif self.dungeon[new_pos[2]][new_pos[0]][new_pos[1]] == 'E':
                self.agent_pos = new_pos
                reward = -10
            else:
                reward = -0.1
        else:
            reward = -0.1

        done = self.agent_pos[:2] == self.goal_pos[:2]
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
        q_values = self.q_table.get(state, [0]*len(self.actions))
        return max(range(len(self.actions)), key=lambda a: q_values[a])

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.setdefault(state, [0]*len(self.actions))
        next_q = self.q_table.setdefault(next_state, [0]*len(self.actions))
        current_q[action] += self.alpha * (reward + self.gamma * max(next_q) - current_q[action])
        self.q_table[state] = current_q
