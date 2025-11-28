import os, math, random
import numpy as np
import pybullet as p
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RobotEnv:

    def __init__(self, gui: bool = True, max_steps: int = 300):
        self.gui = gui
        self.max_steps = max_steps

        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # Масштабы нормализации
        self.pos_scale = 5.0
        self.dist_scale = 5.0
        self.wheel_radius = 0.15     # радиус боковых шаров-колёс (крупные)25 было
        self.body_radius = 0.21       # радиус центрального шара (поменьше)
        self.wheel_base = 0.35        # расстояние между центрами колёс (по Y)

        # "Скоростные" параметры (макс. угловая скорость условных колёс)
        self.max_wheel_velocity = 10.0  # рад/с было 6
        self.sim_dt = 1.0 / 60.0

        # Объекты сцены
        self.ground_id = None
        self.robot_id = None
        self.target_id = None

        # Состояние робота (x, y, yaw)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Для награды
        self.prev_distance = None
        self.step_count = 0

        self.reset()

    def reset(self):
        self._load_scene()
        obs, _ = self._compute_obs()
        return obs

    def close(self):
        if self.client is not None:
            p.disconnect(physicsClientId=self.client)
            self.client = None

    def render(self):
        return

    def _load_scene(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        #КРУТЕЙШИЙ ПОЛ
        ground_half = [10.0, 10.0, 0.1]
        ground_col = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=ground_half,
            physicsClientId=self.client
        )
        ground_vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=ground_half,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            physicsClientId=self.client
        )
        self.ground_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ground_col,
            baseVisualShapeIndex=ground_vis,
            basePosition=[0.0, 0.0, -ground_half[2]],
            physicsClientId=self.client
        )

        self._create_three_sphere_robot()

        tx = float(np.random.uniform(2.0, 4.0) * np.random.choice([-1, 1]))
        ty = float(np.random.uniform(2.0, 4.0) * np.random.choice([-1, 1]))
        tz = 0.25

        col = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.25,
            physicsClientId=self.client
        )
        vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.25,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.client
        )
        self.target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[tx, ty, tz],
            physicsClientId=self.client
        )

        self.prev_distance = self._compute_distance_to_target()
        self.step_count = 0

    def _create_three_sphere_robot(self):
        import pybullet_data

        # чтобы PyBullet находил стандартные модели
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # начальное состояние такое же, как раньше
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # можно взять высоту как раньше через радиус колеса, либо просто 0.2
        base_z = self.wheel_radius  # или base_z = 0.2
        start_pos = [self.x, self.y, base_z]
        start_orn = p.getQuaternionFromEuler([0, 0, self.yaw])

        # грузим готовую машинку из pybullet_data
        self.robot_id = p.loadURDF(
            "racecar/racecar.urdf",
            start_pos,
            start_orn,
            physicsClientId=self.client
        )

        # --- остальное оставляем как было: трение и т.п. ---

        # базовое трение
        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=0.8,
            rollingFriction=0.0,
            spinningFriction=0.0,
            physicsClientId=self.client,
        )

        # для всех сочленений тоже чуть трения
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for j in range(num_joints):
            p.changeDynamics(
                self.robot_id,
                j,
                lateralFriction=0.8,
                rollingFriction=0.0,
                spinningFriction=0.0,
                physicsClientId=self.client,
            )

        # Немного трения для базы и колёс
        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=0.8,
            rollingFriction=0.0,
            spinningFriction=0.0,
            physicsClientId=self.client,
        )
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            p.changeDynamics(
                self.robot_id,
                j,
                lateralFriction=0.8,
                rollingFriction=0.0,
                spinningFriction=0.0,
                physicsClientId=self.client,
            )


    def _get_target_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.target_id, physicsClientId=self.client)
        return np.array(pos, dtype=np.float32)

    def _compute_distance_to_target(self):
        tx, ty, tz = self._get_target_position()
        dx = tx - self.x
        dy = ty - self.y
        return float(math.sqrt(dx * dx + dy * dy))

    def _compute_obs(self):
        tx, ty, tz = self._get_target_position()
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        obs = np.array([
            self.x / self.pos_scale,
            self.y / self.pos_scale,
            math.sin(self.yaw),
            math.cos(self.yaw),
            dx / self.pos_scale,
            dy / self.pos_scale,
            dist / self.dist_scale
        ], dtype=np.float32)

        return obs, float(dist)

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] != 2:
            raise ValueError(f"Ожидалось действие размерности 2, получено {action.shape}")

        a_left = float(np.clip(action[0], -1.0, 1.0))
        a_right = float(np.clip(action[1], -1.0, 1.0))

        v_l = a_left * self.max_wheel_velocity
        v_r = a_right * self.max_wheel_velocity

        r = self.wheel_radius
        half_base = self.wheel_base * 0.5
        v = r * (v_r + v_l) / 2.0
        omega = r * (v_r - v_l) / (2.0 * half_base + 1e-6)

        dt = self.sim_dt
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += omega * dt

        # Обновляем позу в PyBullet
        base_z = 0.04
        base_pos = [self.x, self.y, base_z]
        base_orn = p.getQuaternionFromEuler([0, 0, self.yaw])
        p.resetBasePositionAndOrientation(
            self.robot_id,
            base_pos,
            base_orn,
            physicsClientId=self.client
        )

        # Шаг симуляции (чисто чтобы GUI обновлялся)
        p.stepSimulation(physicsClientId=self.client)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Новое наблюдение
        obs, dist_new = self._compute_obs()

        # --------- НАГРАДА ---------
        prev_dist = self.prev_distance
        self.prev_distance = dist_new

        progress = prev_dist - dist_new   # >0, если приблизились
        reward = 8.0 * progress
        reward -= 0.01

        # Штраф за АФК: большая разница между скоростями колёс (крутится)
        turn_penalty = abs(v_l - v_r)
        reward -= 0.0005 * turn_penalty

        done = False
        info = {}

        if dist_new - prev_dist > 0.03:
            reward -= 0.5


        if dist_new < 0.4:
            reward += 50.0
            done = True
            info["success"] = True
        else:
            info["success"] = False

        if self.step_count >= self.max_steps:
            done = True

        return obs, float(reward), done, info

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float32, device=device)
        action = torch.tensor(np.array(action, dtype=np.float32), dtype=torch.float32, device=device)
        reward = torch.tensor(np.array(reward, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(-1)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32), dtype=torch.float32, device=device)
        done = torch.tensor(np.array(done, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(-1)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256),
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = MLP(state_dim, 2 * action_dim, hidden_sizes)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)  # в диапазоне [-1, 1]

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = MLP(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        target_update_interval=1,
        automatic_entropy_tuning=True,
    ):
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Политика (actor)
        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        # Две Q-сети (критики)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        # Автоматическая настройка температуры (энтропии)
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, mean_action = self.policy.sample(state_t)
                action = mean_action
            else:
                action, _, _ = self.policy.sample(state_t)
        return action.cpu().numpy()[0]

    def soft_update(self, source, target):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def update_parameters(self, memory: ReplayBuffer, batch_size: int, updates: int):
        if len(memory) < batch_size:
            return

        state, action, reward, next_state, done = memory.sample(batch_size)

        # ---------- Обновление критиков ----------
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)
            q_next_min = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = reward + (1 - done) * self.gamma * q_next_min

        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = nn.MSELoss()(q1_pred, q_target)
        q2_loss = nn.MSELoss()(q2_pred, q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # ---------- Обновление политики ----------
        new_action, log_pi, _ = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_min_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_pi - q_min_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ---------- Обновление температуры энтропии ----------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # ---------- Мягкое обновление target-сетей ----------
        if updates % self.target_update_interval == 0:
            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)

def train_sac(
    num_episodes=400,
    max_steps=300,
    batch_size=256,
    start_steps=2000,
    updates_per_step=1,
    replay_capacity=200_000,
    live_render=True,
    render_from_episode=1,
):

    use_gui_now = live_render and (render_from_episode <= 1)
    env = RobotEnv(gui=use_gui_now, max_steps=max_steps)

    state_dim = 7
    action_dim = 2

    agent = SACAgent(state_dim, action_dim)
    memory = ReplayBuffer(capacity=replay_capacity)

    rewards_history = []
    avg_rewards_history = []

    total_env_steps = 0
    updates = 0

    for episode in range(1, num_episodes + 1):
        # Если пришло время включить GUI — пересоздаём env с gui=True
        if live_render and (episode == render_from_episode) and not env.gui:
            env.close()
            env = RobotEnv(gui=True, max_steps=max_steps)
            print(f"\n=== ВклЮчиЛ ВИЗУАЛИЗАЦИЮ С ЭПИЗОДА {episode} ===\n")

        state = env.reset()
        episode_reward = 0.0
        episode_success = False  # была ли достигнута цель в этом эпизоде

        for t in range(max_steps):
            if total_env_steps < start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)

            # Фиксируем, был ли успех
            if info.get("success", False):
                episode_success = True

            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_env_steps += 1

            if total_env_steps >= start_steps:
                for _ in range(updates_per_step):
                    agent.update_parameters(memory, batch_size, updates)
                    updates += 1

            if done:
                break

        rewards_history.append(episode_reward)
        avg_reward = float(np.mean(rewards_history[-50:]))
        avg_rewards_history.append(avg_reward)

        mark = "✅" if episode_success else "  "

        print(
            f"{mark} Эпизод {episode}/{num_episodes} | "
            f"Награда: {episode_reward:.2f} | "
            f"MovingAvg(50): {avg_reward:.2f}"
        )

    env.close()

    plt.figure()
    plt.plot(rewards_history, label="Reward per episode")
    plt.plot(avg_rewards_history, label="Moving average (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("telega_sac_rewards.png")
    plt.close()

    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
        },
        "../../Downloads/telega_sac_agent.pt",
    )

    print("=== Обучение завершено ===")
    print("Модель сохранена в telega_sac_agent.pt")
    print("График наград сохранён в telega_sac_rewards.png")

if __name__ == "__main__":
    train_sac(
        num_episodes=1000,
        max_steps=300,
        batch_size=256,
        start_steps=2000,
        updates_per_step=1,
        replay_capacity=200_000,
        live_render=True,
        render_from_episode=800,
    )
