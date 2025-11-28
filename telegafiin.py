import time
import torch
from telega import RobotVisionEnv, VisionSACAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_trained(model_path="telega_pixel_agent.pt", max_steps=300, episodes=10):
    print(f"Загружаю модель {model_path} (пиксельная политика)...")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    env = RobotVisionEnv(gui=True, max_steps=max_steps)
    state = env.reset()

    vision_channels = state["vision"].shape[0]
    proprio_dim = state["proprio"].shape[0]
    action_dim = 2
    agent = VisionSACAgent(vision_channels, proprio_dim, action_dim)

    agent.policy.load_state_dict(checkpoint["policy"])

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        episode_success = False

        for _ in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if info.get("success", False):
                episode_success = True

            time.sleep(env.sim_dt)

            if done:
                break

        mark = "+" if episode_success else "-"
        print(f"{mark} Эпизод {ep}/{episodes}, награда: {total_reward:.2f}")

    env.close()
    print("Демо завершено.")


if __name__ == "__main__":
    visualize_trained(episodes=10)
