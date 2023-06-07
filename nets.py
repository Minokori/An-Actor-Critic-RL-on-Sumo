# region 库导入
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sumoenv
import collections
import random
# endregion

# region 全局变量
RETURNLIST:list = []
"""奖励"""
# endregion

class ReplayBuffer():
    """经验回放池类
    """

    def __init__(self, capacity: int):
        """经验回放池类 , 最大存储序列数为 `capacity`

        ---
        每一条数据为 `(state , action , reward , next_state , done)`

        ---
        Args:
            capacity (int): 最大存储序列数
        """
        self.buffer = collections.deque(maxlen=capacity)
        """
        双端队列 , 类似`list`. `maxlen`限定最大长度.
        元素已满且有新元素从一端 “入队” 时 , 数量相同的旧元素将从另一端 “出队” (被移除)
        """

    def add(self, state, action, reward, next_state, done: bool) -> None:
        """向池中以元组 `tuple` 形式添加一条序列 `(s,a,r,s')`

        ---
        Args:
            state (Iterable): 以 可迭代数据类型 表示的状态 `s` , e.g. : `list` , `dict`\n
            action (Number): 以 数字类型 表示的动作 `a` , e.g. : `int`\n
            reward (Number): 以 数字类型 表示的动作回报 `r` , e.g. : `int` , `float`\n
            next_state (Iterable): 以 可迭代数据类型 表示的状态 `s'` , e.g. : `list` , `dict`\n
            done (bool): 是否达到终止状态
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, tuple, tuple, np.ndarray, tuple]:
        """从池中以元组 `tuple` 形式随机取出 `batch_size` 条序列 `(s,a,r,s')`\n
        并将各属性以一个可迭代对象的方式返回

        Args:
            batch_size (int): 要取出序列的数量

        Returns:
            tuple: 元组 , `(state , action , reward , next_state , done)`\n
            * `state` : batchsize * statedim
            * `action` : batchsize * actiondim
            * `reward` : batchsize * 1
            * `next_state` : batchsize * statedim
            * `done` : batchsize * 1
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self) -> int:
        """返回池中序列的数量

        Returns:
            int: 池中序列的数量
        """
        return len(self.buffer)

# region 定义策略网络 PolicyNet
class PolicyNet(torch.nn.Module):
    """#### 策略网络

    ---
    输入是某个状态 `s`

    输出是该状态下的动作概率分布 `P(a|s)`
    
    ---
    采用在离散动作空间上的 `softmax()` 实现可学习的多项分布
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> torch.nn.Module:
        """只有一层隐藏层的网络

        ---
        网络结构: `fc1(Linear)` --> `relu` --> `fc2(Linear)` --> `softmax`\n

        Args:
            state_dim (int): 环境状态 `s` 的维度\n
            hidden_dim (int): 隐藏层 维度\n
            action_dim (int): 动作空间 `A` 的维度\n
        """
        super(PolicyNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        """
        网络第一个全连接层.\n
        输入维度为 环境状态 `s` 的维度\n
        输出维度为 隐藏层 维度
        """
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        """
        网络第二个全连接层.\n
        输入维度为 隐藏层 维度\n
        输出维度为 动作空间 `A` 的维度
        """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """网络前向传播

        Args:
            x (torch.Tensor): 输入网络的样本

        Returns:
            torch.Tensor: 网络输出 P(a|s)
        """
        x = F.elu(self.fc1(x),1)
        x = F.elu(self.fc2(x),1)
        x = F.softmax(x,dim=1)
        return x
# endregion

# region 定义价值网络 ValueNet
class ValueNet(torch.nn.Module):
    """价值网络

    ---
    输入是某个状态 `s`

    输出是该状态的动作价值 `Q(a|s)`
    """
    def __init__(self, state_dim:int, hidden_dim:int):
        """只有一层隐藏层的网络

        ---
        网络结构: `fc1(Linear)` --> `relu` --> `fc2(Linear)`

        Args:
            state_dim (int): 环境状态 `s` 的维度
            hidden_dim (int): 隐藏层 维度
            action_dim (int): 动作空间 `A` 的维度
        """
        super(ValueNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        """
        网络第一个全连接层.\n
        输入维度为 环境状态 `s` 的维度\n
        输出维度为 隐藏层 维度
        """
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        """
        网络第二个全连接层.\n
        输入维度为 隐藏层 维度\n
        输出维度为 动作空间 `A` 的维度
        """
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """网络前向传播

        Args:
            x (torch.Tensor): 输入网络的样本

        Returns:
            torch.Tensor: 网络输出 Q(a|s)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)
# endregion


# region 定义ActorCritic算法
class ActorCritic():
    """Actor-Critic算法
    """
    def __init__(self, state_dim:int, hidden_dim:int, action_dim:int, actor_lr:int, critic_lr:int, gamma:float, device:torch.device):
        """Actor-Critic算法

        Args:
            state_dim (int): 网络输入维数 , 即环境状态 s 的维数\n
            hidden_dim (int): 网络隐藏层维数\n
            action_dim (int): 网络输出层维数 , 即动作空间A的维数\n
            actor_lr (int): 策略网络的学习率\n
            critic_lr (int): 价值网络的学习率\n
            gamma (float): 折扣因子\n
            device (torch.device): 训练设备\n
        """
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        """策略网络"""
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        """价值网络"""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        """策略网络优化器"""
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        """价值网络优化器"""
        self.gamma = gamma
        """折扣因子"""
        self.device = device
        """训练设备"""

    def take_action(self, state)->int:
        """根据状态采取动作

        Args:
            state (torch.Tensor): 状态张量

        Returns:
            tuple[int,torch.Tensor]: 采取动作的索引，动作值
        """
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs:torch.Tensor = self.actor(state)
        probs = torch.where(torch.isnan(probs), torch.full_like(probs, 0), probs)
        action_dist = torch.distributions.Categorical(probs) # 动作概率分布
        action:torch.Tensor = action_dist.sample() # 抽取一个动作的编号
        return action.item() # 返回这个 int 编号

    def update(self, transition_dict:dict):
        """更新网络

        Args:
            transition_dict (dict): 状态转移链
                `keys`:\n
                        - "states" : 状态 List
                        - "actions" : 动作 List
                        - "rewards" : 奖励 List
                        - "next_states" : 下一状态 List
                        - "dones" : 仿真完成标志位 List
        """
        torch.autograd.set_detect_anomaly(True)
        # 将 状态转移链移到 GPU 上
        states = torch.tensor(np.array(transition_dict['states']),dtype=torch.float).to(self.device)# shape = (轨迹条数, states长度)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) # shape= (轨迹数,1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device) # 同state
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device) # shpae = (轨迹个数,1)

        # 时序差分目标
        td_target:torch.Tensor = rewards + self.gamma * self.critic(next_states) * (1 - dones) # shape = (轨迹,1)
        
        # 时序差分误差
        td_delta:torch.Tensor = td_target - self.critic(states) # shape = 同上
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            actor_loss.backward()  # 计算策略网络的梯度
            critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
# endregion

# region 训练函数
def train_on_policy_agent(env:sumoenv.SumoEnv, agent:ActorCritic, num_episodes:int)->list[float]:
    """在线训练

    Args:
        env (sumoenv.SumoEnv): `sumo` 环境
        agent (ActorCritic): AC网络
        num_episodes (int): 训练步数

    Returns:
        list[float]: 每步的奖励
    """
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [],
                                   'next_states': [], 'rewards': [], 'dones': []}
                state:dict = env.reset()
                state:np.ndarray = sumoenv.GetFlattenedState(state)
                done = False
                while not done:
                    action:int = agent.take_action(state) # 相位编号
                    action_phase = action % 4 * 3
                    next_state, reward, done, truncated, info = env.step(action_phase)

                    next_state:np.ndarray = sumoenv.GetFlattenedState(next_state)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (
                        num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env:sumoenv.SumoEnv, agent:ActorCritic, num_episodes:int, replay_buffer:ReplayBuffer, minimal_size:int, batch_size:int)->list[float]:
    """离线训练

    Args:
        env (sumoenv.SumoEnv): `sumo` 环境
        agent (ActorCritic): AC网络
        num_episodes (int): 训练步数
        replay_buffer (ReplayBuffer): 经验回放池
        minimal_size (int): 经验回放池最小限制容量
        batch_size (int): 每次从经验回放池采样的状态转移链条数
    Returns:
        list[float]: 每步的奖励
    """
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                
                state:dict = env.reset()
                state:np.ndarray = sumoenv.GetFlattenedState(state)

                done = False

                while not done:
                    action:int = agent.take_action(state)
                    action_phase = action % 4 * 3
                    next_state, reward, done, truncated, info = env.step(action_phase)
                    next_state:np.ndarray = sumoenv.GetFlattenedState(next_state)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                global RETURNLIST
                RETURNLIST.append(episode_return)
                return_list.append(episode_return)
                
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                
                pbar.update(1)
    return return_list
# endregion