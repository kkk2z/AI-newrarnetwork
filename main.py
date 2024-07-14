import time
from dungeon import create_dungeon
from agent import DungeonEnv, QLearningAgent

def print_dungeon(dungeon, agent_pos):
    for layer in range(len(dungeon)):
        print(f"Layer {layer}:")
        for i, row in enumerate(dungeon[layer]):
            line = ''.join(row)
            if agent_pos[2] == layer and agent_pos[0] == i:
                line = line[:agent_pos[1]] + 'A' + line[agent_pos[1] + 1:]  # エージェントを'A'で表示
            print(line)
        print("\n")

if __name__ == "__main__":
    layers = 3  # 階層数
    dungeon = create_dungeon(10, 10, layers)
    env = DungeonEnv(dungeon)
    agent = QLearningAgent(actions=[0, 1, 2, 3, 4, 5])  # 上下移動のアクションを追加

    for episode in range(10):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            print_dungeon(dungeon, env.agent_pos)
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            steps += 1
            time.sleep(0.5)

    print("トレーニングが完了しました。")
