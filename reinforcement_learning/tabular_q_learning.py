import numpy as np
import pandas as pd
import time

np.random.seed(2)

# 一维世界的宽度
N_STATES = 6
# 探索者的可用动作
ACTIONS = ['left', 'right']
# 贪婪度 greedy
EPSILON = 0.9
# 学习率
ALPHA = 0.1
# 奖励递减值
GAMMA = 0.9
# 最大回合数
MAX_EPISODES = 13
# 移动间隔时间
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    # 构建n_states行，2列的全0矩阵 存储Q值
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions, )
    return table


def choose_action(state, q_table):
    # 选出这个state的所有action值 即某一行的所有值
    state_actions = q_table.iloc[state, :]
    # 非贪婪或者该行的state还没有探索过
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        # 贪婪模式(选择最大值的索引)
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    # agent 如何同环境进行交互
    if A == 'right':
        if S == N_STATES - 2:  # 到达reward点
            _S = 'terminal'
            R = 1  # 奖励reward = 1
        else:
            _S = S + 1  # 右移
            R = 0
    else:
        R = 0
        if S == 0:
            _S = S  # 撞墙了
        else:
            _S = S - 1
    return _S, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode {0}: total_steps = {1}'.format(episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                             ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始化q table
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0  # 回合初始位置
        is_terminated = False  # 是否回合结束
        update_env(S, episode, step_counter)  # 环境更新
        while not is_terminated:
            A = choose_action(S, q_table)  # 选择行为
            _S, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]  # 估算的(状态-行为)值
            if _S != 'terminal':
                q_target = R + GAMMA * q_table.iloc[_S, :].max()  # 实际的(状态-行为)值 (回合未结束)
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 更新q_table 类似于nn的反向传播
            S = _S  # 探索者移动到下一个state

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-Table:\n')
    print(q_table)
