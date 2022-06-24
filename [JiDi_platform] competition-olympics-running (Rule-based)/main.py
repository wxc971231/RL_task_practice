import sys
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(father_path))
from olympics.generator import create_scenario
from olympics.scenario.running import Running
import argparse

from agents.my.submission import agent as my_agent
from agents.champion_raw.submission import agent as champion_agent

import random
import numpy as np
import json

def store(record, name):
    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="map4", type=str,
                        help= "map1/map2/map3/map4")
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    #agent1 = random_agent()
    #agent2 = random_agent()

    #map_index_seq = list(range(1,5))
    #map_index_seq = list(range(1,12))
    
    my_win = np.zeros(12,dtype = np.int)
    my_tie = np.zeros(12,dtype = np.int)

    for i in range(100):
        for ind in range(11):
            print("==========================================")

            my_agent.reset()
            champion_agent.reset()

            ind += 1
            print("map index: ", ind)
            Gamemap = create_scenario("map"+str(ind))
            #map_index_seq.append(ind)

            rnd_seed = random.randint(0, 1000)
            game = Running(Gamemap, seed = rnd_seed)
            game.map_num = ind

            obs = game.reset()
            if RENDER:
                game.render()

            done = False
            step = 0
            if RENDER:
                game.render('MAP {}'.format(ind))

            while not done:
                step += 1

                #action1 = agent1.act(obs[0])
                action1 = champion_agent.choose_action({'obs':obs[0]})
                action2 = my_agent.choose_action({'obs':obs[1]})

                action1 = [action1[0][0],action1[1][0]]
                action2 = [action2[0][0],action2[1][0]]

                obs, reward, done, _ = game.step([action1, action2])

                if RENDER:
                    game.render()

            if reward[1] == 100:
                if reward[0] == 0:
                    my_win[ind] += 1
                else:
                    my_tie[ind] += 1

            print('Episode Reward = {}'.format(reward))

        print('*********************')
        print('第{}轮对抗'.format(i+1))
        print(my_win/(i+1))
        print(my_tie/(i+1))
