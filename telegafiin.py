import time
import torch
from telega import RobotEnv, SACAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_trained(max_steps=300, episodes=30):
    print("Загружаю модель telega_sac_agent.pt (готовую модель)...")
    checkpoint = torch.load("telega_sac_agent.pt", map_location=DEVICE)

    env = RobotEnv(gui=True, max_steps=max_steps)


    state_dim = 7
    action_dim = 2
    agent = SACAgent(state_dim, action_dim)

    agent.policy.load_state_dict(checkpoint["policy"])

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        episode_success = False

        for t in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if info.get("success", False):
                episode_success = True

            # чуть притормозить, чтобы глазом успевать
            time.sleep(env.sim_dt)

            if done:
                break

        mark = "+" if episode_success else "-"
        print(f"{mark} Эпизод {ep}/{episodes}, награда: {total_reward:.2f}")

    env.close()
    print("Демо завершено.")


if __name__ == "__main__":
    visualize_trained(episodes=30)
