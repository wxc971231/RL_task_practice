import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)


from collections import deque, namedtuple


from env.chooseenv import make
from rl_trainer.log_path import *
from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.random import random_agent


parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-running", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=1500, type=int)
parser.add_argument('--episode_length', default=500, type=int)
parser.add_argument('--map', default=1, type = int)
parser.add_argument('--shuffle_map', action='store_true')

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=900, type=int)


device = 'cpu'
RENDER = True
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name)
    if not args.shuffle_map:
        env.specify_a_map(args.map)         #specifying a map, you can also shuffle the map by not doing this step

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    ctrl_agent_index = 1
    print(f'Agent control by the actor: {ctrl_agent_index}')

    ctrl_agent_num = 1

    width = env.env_core.view_setting['width']+2*env.env_core.view_setting['edge']
    height = env.env_core.view_setting['height']+2*env.env_core.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = 25*25
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on map {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args.algo, 'all' if args.shuffle_map else args.map)))
        save_config(args, log_dir)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    if args.load_model:
        model = PPO()
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir,episode=args.load_episode)
    else:
        model = PPO(run_dir)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    opponent_agent = random_agent()     #we use random opponent agent here

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        state = env.reset(args.shuffle_map)    #[{'obs':[25,25], "control_player_index": 0}, {'obs':[25,25], "control_player_index": 1}]
        if RENDER:
            env.env_core.render()
        obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs']).flatten()     #[25*25]
        obs_oppo_agent = state[1-ctrl_agent_index]['obs']   #[25,25]

        episode += 1
        step = 0
        Gt = 0

        while True:
            action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
            action_opponent = [[0],[0]]  #here we assume the opponent is not moving in the demo

            action_ctrl_raw, action_prob= model.select_action(obs_ctrl_agent, False if args.load_model else True)
                            #inference
            action_ctrl = actions_map[action_ctrl_raw]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]        #wrapping up the action

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

            next_state, reward, done, _, info = env.step(action)

            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1-ctrl_agent_index]['obs']

            step += 1

            if not done:
                post_reward = [-1., -1.]
            else:
                if reward[0] != reward[1]:
                    post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                else:
                    post_reward=[-1., -1.]

            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)

            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
            if RENDER:
                env.env_core.render()
            Gt += reward[ctrl_agent_index] if done else -1

            if done:
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                      "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)

                if not args.load_model:
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        if win_is == 1:
                            model.update(episode)
                            train_count += 1
                        else:
                            model.clear_buffer()

                    writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)





if __name__ == '__main__':
    args = parser.parse_args()
    #args.load_model = True
    #args.load_run = 3
    #args.map = 3
    #args.load_episode= 900
    main(args)