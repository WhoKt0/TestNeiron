import os, math, random
from typing import Optional

import numpy as np
import pybullet as p
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt


# =======================
#  Общие настройки
# =======================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _grayscale_from_rgb(rgb_array: np.ndarray) -> np.ndarray:
    """Преобразование RGB [H, W, 3] в градации серого [H, W]."""
    r, g, b = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

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
        self.no_progress_steps = 0

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

        # стандартный плоский пол из pybullet_data
        import pybullet_data

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

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
        self.no_progress_steps = 0

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

    def _compute_reward(self, progress: float, dist_new: float, prev_dist: float, turning_gap: float):
        # Денс за движение к цели
        reward = 6.0 * progress

        # Маленький штраф за время
        reward -= 0.01

        # Накапливаем штраф за отсутствие прогресса (борьба с "застрял и повторяю")
        if progress < 0.005:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        reward -= 0.02 * min(self.no_progress_steps, 50)

        # Штраф за заметное удаление от цели
        if dist_new > prev_dist + 0.02:
            reward -= 0.4

        # Лёгкий штраф за сильное вращение на месте
        reward -= 0.0007 * turning_gap

        done = False
        info = {"success": False}

        # Финальный бонус за достижение цели
        if dist_new < 0.35:
            reward += 60.0
            done = True
            info["success"] = True

        if self.step_count >= self.max_steps:
            done = True

        return float(reward), done, info

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

        prev_dist = self.prev_distance
        self.prev_distance = dist_new

        progress = prev_dist - dist_new   # >0, если приблизились
        reward, done, info = self._compute_reward(progress, dist_new, prev_dist, abs(v_l - v_r))

        return obs, reward, done, info


class RobotVisionEnv(RobotEnv):
    """Среда с визуальным наблюдением: стек кадров + простая проприоцепция.

    Наблюдение:
        {
            "vision": np.ndarray [frame_stack, H, W] (grayscale, 0..1),
            "proprio": np.ndarray [2] => sin(yaw), cos(yaw)
        }

    Действия: аналогичны RobotEnv (2 значения в диапазоне [-1, 1]).
    """

    def __init__(
        self,
        gui: bool = True,
        max_steps: int = 300,
        camera_width: int = 64,
        camera_height: int = 64,
        frame_stack: int = 4,
        frame_skip: int = 4,
        grayscale: bool = True,
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)

        super().__init__(gui=gui, max_steps=max_steps)

    def reset(self):
        self._load_scene()
        first_frame = self._render_camera()
        self._init_frame_stack(first_frame)
        self._perform_initial_spin()
        self.prev_distance = self._compute_distance_to_target()
        self.no_progress_steps = 0
        obs, _ = self._compute_obs(latest_frame=None, append=False)
        return obs

    def _init_frame_stack(self, first_frame: np.ndarray):
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(first_frame)

    def _render_camera(self) -> np.ndarray:
        cam_target = [self.x, self.y, 0.0]
        cam_distance = 3.5
        cam_height = 3.5
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[self.x, self.y - cam_distance, cam_height],
            cameraTargetPosition=cam_target,
            cameraUpVector=[0, 0, 1],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=10.0,
        )

        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        width, height, rgba, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
            physicsClientId=self.client,
        )

        rgb = np.reshape(rgba, (height, width, 4))[..., :3].astype(np.float32)
        if self.grayscale:
            frame = _grayscale_from_rgb(rgb) / 255.0
        else:
            frame = rgb / 255.0
        return frame.astype(np.float32)

    def _perform_initial_spin(self):
        """Повернуть камеру на 360° перед тем, как агент начнёт действовать."""
        spin_steps = 24
        start_yaw = self.yaw
        base_z = 0.04

        for i in range(spin_steps):
            angle = start_yaw + 2 * math.pi * (i + 1) / spin_steps
            self.yaw = angle
            base_pos = [self.x, self.y, base_z]
            base_orn = p.getQuaternionFromEuler([0, 0, self.yaw])
            p.resetBasePositionAndOrientation(
                self.robot_id,
                base_pos,
                base_orn,
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)
            self.frames.append(self._render_camera())

        # Вернуться к исходной ориентации
        self.yaw = start_yaw
        base_pos = [self.x, self.y, base_z]
        base_orn = p.getQuaternionFromEuler([0, 0, self.yaw])
        p.resetBasePositionAndOrientation(
            self.robot_id,
            base_pos,
            base_orn,
            physicsClientId=self.client,
        )
        p.stepSimulation(physicsClientId=self.client)
        self.frames.append(self._render_camera())

    def _compute_obs(self, latest_frame: Optional[np.ndarray] = None, append: bool = True):
        frame = latest_frame if latest_frame is not None else self._render_camera()
        if append:
            self.frames.append(frame)
        stacked = np.stack(self.frames, axis=0)
        proprio = np.array([
            math.sin(self.yaw),
            math.cos(self.yaw),
        ], dtype=np.float32)

        dist = self._compute_distance_to_target()
        obs = {"vision": stacked.astype(np.float32), "proprio": proprio}
        return obs, float(dist)

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] != 2:
            raise ValueError(f"Ожидалось действие размерности 2, получено {action.shape}")

        a_left = float(np.clip(action[0], -1.0, 1.0))
        a_right = float(np.clip(action[1], -1.0, 1.0))

        r = self.wheel_radius
        half_base = self.wheel_base * 0.5

        # Frame skipping: повторяем действие несколько тиков
        for _ in range(self.frame_skip):
            v_l = a_left * self.max_wheel_velocity
            v_r = a_right * self.max_wheel_velocity
            v = r * (v_r + v_l) / 2.0
            omega = r * (v_r - v_l) / (2.0 * half_base + 1e-6)

            dt = self.sim_dt
            self.x += v * math.cos(self.yaw) * dt
            self.y += v * math.sin(self.yaw) * dt
            self.yaw += omega * dt

            base_z = 0.04
            base_pos = [self.x, self.y, base_z]
            base_orn = p.getQuaternionFromEuler([0, 0, self.yaw])
            p.resetBasePositionAndOrientation(
                self.robot_id,
                base_pos,
                base_orn,
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)

        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        new_frame = self._render_camera()
        obs, dist_new = self._compute_obs(latest_frame=new_frame, append=True)

        prev_dist = self.prev_distance
        self.prev_distance = dist_new

        progress = prev_dist - dist_new
        reward, done, info = self._compute_reward(progress, dist_new, prev_dist, abs(a_left - a_right))

        return obs, reward, done, info

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


class VisionReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append({
            "state": {
                "vision": np.array(state["vision"], dtype=np.float32, copy=True),
                "proprio": np.array(state["proprio"], dtype=np.float32, copy=True),
            },
            "action": np.array(action, dtype=np.float32, copy=True),
            "reward": float(reward),
            "next_state": {
                "vision": np.array(next_state["vision"], dtype=np.float32, copy=True),
                "proprio": np.array(next_state["proprio"], dtype=np.float32, copy=True),
            },
            "done": float(done),
        })

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        vision = np.stack([b["state"]["vision"] for b in batch], axis=0)
        proprio = np.stack([b["state"]["proprio"] for b in batch], axis=0)
        next_vision = np.stack([b["next_state"]["vision"] for b in batch], axis=0)
        next_proprio = np.stack([b["next_state"]["proprio"] for b in batch], axis=0)

        state = {
            "vision": torch.tensor(vision, dtype=torch.float32, device=device),
            "proprio": torch.tensor(proprio, dtype=torch.float32, device=device),
        }
        next_state = {
            "vision": torch.tensor(next_vision, dtype=torch.float32, device=device),
            "proprio": torch.tensor(next_proprio, dtype=torch.float32, device=device),
        }

        action_np = np.stack([b["action"] for b in batch], axis=0)
        action = torch.tensor(action_np, dtype=torch.float32, device=device)
        reward = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        done = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=device).unsqueeze(-1)

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


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int, proprio_dim: int, feature_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # фиксируем выход фичей в 4x4 даже для других размеров кадра
        self.fc = nn.Sequential(
            nn.Linear(1024 + proprio_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, vision, proprio):
        conv_features = self.cnn(vision)
        x = torch.cat([conv_features, proprio], dim=-1)
        return self.fc(x)


class VisionGaussianPolicy(nn.Module):
    def __init__(self, vision_channels: int, proprio_dim: int, action_dim: int,
                 feature_dim: int = 256, hidden_sizes=(256, 256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.encoder = VisionEncoder(vision_channels, proprio_dim, feature_dim)
        self.mlp = MLP(feature_dim, 2 * action_dim, hidden_sizes)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, vision, proprio):
        features = self.encoder(vision, proprio)
        x = self.mlp(features)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, vision, proprio):
        mean, log_std = self.forward(vision, proprio)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class VisionQNetwork(nn.Module):
    def __init__(self, vision_channels: int, proprio_dim: int, action_dim: int,
                 feature_dim: int = 256, hidden_sizes=(256, 256)):
        super().__init__()
        self.encoder = VisionEncoder(vision_channels, proprio_dim, feature_dim)
        self.head = MLP(feature_dim + action_dim, 1, hidden_sizes)

    def forward(self, vision, proprio, action):
        features = self.encoder(vision, proprio)
        x = torch.cat([features, action], dim=-1)
        return self.head(x)


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


class VisionSACAgent:
    def __init__(
        self,
        vision_channels: int,
        proprio_dim: int,
        action_dim: int,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        target_update_interval=1,
        automatic_entropy_tuning=True,
        feature_dim=256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.policy = VisionGaussianPolicy(
            vision_channels=vision_channels,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
        ).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        self.q1 = VisionQNetwork(vision_channels, proprio_dim, action_dim, feature_dim=feature_dim).to(device)
        self.q2 = VisionQNetwork(vision_channels, proprio_dim, action_dim, feature_dim=feature_dim).to(device)
        self.q1_target = VisionQNetwork(vision_channels, proprio_dim, action_dim, feature_dim=feature_dim).to(device)
        self.q2_target = VisionQNetwork(vision_channels, proprio_dim, action_dim, feature_dim=feature_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate: bool = False):
        vision = torch.tensor(state["vision"], dtype=torch.float32, device=device).unsqueeze(0)
        proprio = torch.tensor(state["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, mean_action = self.policy.sample(vision, proprio)
                action = mean_action
            else:
                action, _, _ = self.policy.sample(vision, proprio)
        return action.cpu().numpy()[0]

    def soft_update(self, source, target):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def update_parameters(self, memory: VisionReplayBuffer, batch_size: int, updates: int):
        if len(memory) < batch_size:
            return

        state, action, reward, next_state, done = memory.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state["vision"], next_state["proprio"])
            q1_next = self.q1_target(next_state["vision"], next_state["proprio"], next_action)
            q2_next = self.q2_target(next_state["vision"], next_state["proprio"], next_action)
            q_next_min = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = reward + (1 - done) * self.gamma * q_next_min

        q1_pred = self.q1(state["vision"], state["proprio"], action)
        q2_pred = self.q2(state["vision"], state["proprio"], action)
        q1_loss = nn.MSELoss()(q1_pred, q_target)
        q2_loss = nn.MSELoss()(q2_pred, q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        new_action, log_pi, _ = self.policy.sample(state["vision"], state["proprio"])
        q1_new = self.q1(state["vision"], state["proprio"], new_action)
        q2_new = self.q2(state["vision"], state["proprio"], new_action)
        q_min_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_pi - q_min_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

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


def train_pixel_sac(
    num_episodes=500,
    max_steps=300,
    batch_size=128,
    start_steps=2000,
    updates_per_step=1,
    replay_capacity=100_000,
    live_render=True,
    render_from_episode=1,
    frame_stack=4,
    frame_skip=4,
    camera_size=64,
    model_path="telega_pixel_agent.pt",
):
    use_gui_now = live_render and (render_from_episode <= 1)
    env = RobotVisionEnv(
        gui=use_gui_now,
        max_steps=max_steps,
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        camera_width=camera_size,
        camera_height=camera_size,
    )

    example_state = env.reset()
    vision_channels = example_state["vision"].shape[0]
    proprio_dim = example_state["proprio"].shape[0]
    action_dim = 2

    agent = VisionSACAgent(vision_channels, proprio_dim, action_dim)
    memory = VisionReplayBuffer(capacity=replay_capacity)

    rewards_history = []
    avg_rewards_history = []
    total_env_steps = 0
    updates = 0

    for episode in range(1, num_episodes + 1):
        if live_render and (episode == render_from_episode) and not env.gui:
            env.close()
            env = RobotVisionEnv(
                gui=True,
                max_steps=max_steps,
                frame_stack=frame_stack,
                frame_skip=frame_skip,
                camera_width=camera_size,
                camera_height=camera_size,
            )
            print(f"\n=== Включил визуализацию с эпизода {episode} ===\n")

        state = env.reset()
        episode_reward = 0.0
        episode_success = False

        for t in range(max_steps):
            if total_env_steps < start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done, info = env.step(action)
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
    plt.savefig("telega_pixel_rewards.png")
    plt.close()

    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
        },
        model_path,
    )

    print("=== Обучение на пикселях завершено ===")
    print(f"Модель сохранена в {model_path}")
    print("График наград сохранён в telega_pixel_rewards.png")

if __name__ == "__main__":
    # Для пиксельного обучения используем сжатое изображение + frame stacking/skip.
    train_pixel_sac(
        num_episodes=50,
        max_steps=300,
        batch_size=128,
        start_steps=1000,
        updates_per_step=1,
        replay_capacity=100_000,
        live_render=False,
        render_from_episode=10,
        frame_stack=4,
        frame_skip=4,
        camera_size=64,
    )
