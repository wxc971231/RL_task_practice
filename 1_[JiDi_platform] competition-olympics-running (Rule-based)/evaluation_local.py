
import numpy as np
import torch
import random
from agents.rl.submission import agent as rl_agent
from agents.my.submission import agent as my_agent
from agents.champion_raw.submission import agent as champion_agent
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


def get_join_actions(state, algo_list):

    joint_actions = []

    for agent_idx in range(len(algo_list)):
        if algo_list[agent_idx] == 'random':
            driving_force = random.uniform(-100, 200)
            turing_angle = random.uniform(-30, 30)
            joint_actions.append([[driving_force], [turing_angle]])

        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs'].flatten()
            actions_raw = rl_agent.choose_action(obs)
            actions = actions_map[actions_raw]
            joint_actions.append([[actions[0]], [actions[1]]])

        elif algo_list[agent_idx] == 'my':
            obs = state[agent_idx]
            actions_raw = my_agent.choose_action(obs)
            joint_actions.append(actions_raw)

        elif algo_list[agent_idx] == 'my':
            obs = state[agent_idx]
            actions_raw = my_agent.choose_action(obs)
            joint_actions.append(actions_raw)
    
        elif algo_list[agent_idx] == 'champion_raw':
            obs = state[agent_idx]
            actions_raw = champion_agent.choose_action(obs)
            joint_actions.append(actions_raw)

    return joint_actions

RENDER = True

def run_game(env, algo_list, episode, shuffle_map,map_num, verbose=False):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)       #agent 1 win, agent 2 win, draw
    episode = int(episode)
    for i in range(1, int(episode)+1):
        episode_reward = np.zeros(2)

        state = env.reset(shuffle_map)
        if RENDER:
            env.env_core.render()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if RENDER:
                env.env_core.render()

            if done:
                if reward[0] != reward[1]:
                    if reward[0]==100:
                        num_win[0] +=1
                    elif reward[1] == 100:
                        num_win[1] += 1
                    else:
                        raise NotImplementedError
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i==episode:
                        print()
                break
            state = next_state
            step += 1
        total_reward += episode_reward

        my_agent.reset()
        champion_agent.reset()

    total_reward/=episode
    print("total reward: ", total_reward)
    print('Result in map {} within {} episode:'.format(map_num, episode))
    #print(f'\nResult in base on {episode} in map {map_num} ', end='')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--my_ai", default='rl', help='rl/random')
    #parser.add_argument("--opponent", default='random', help='rl/random')

    parser.add_argument("--my_ai", default="my", help="random/my/champion_raw")                 # 在此选择我方控制的 agent          
    parser.add_argument("--opponent", default="champion_raw", help="random/my/champion_raw")    # 在此选择对方控制的 agent

    parser.add_argument("--episode", default=20)
    parser.add_argument("--map", default='11', help='1/2/3/4/all')
    args = parser.parse_args()

    env_type = "olympics-running"
    game = make(env_type, conf=None, seed = 1)

    if args.map != 'all':
        game.specify_a_map(int(args.map))
        shuffle = False
    else:
        shuffle = True

    #torch.manual_seed(1)
    #np.random.seed(1)
    #random.seed(1)

    agent_list = [args.opponent, args.my_ai]        #your are controlling agent green
    run_game(game, algo_list=agent_list, episode=args.episode, shuffle_map=shuffle,map_num=args.map,verbose=False)

