from core import OlympicsBase
import time

class Running(OlympicsBase):
    def __init__(self, map, seed = None):
        super(Running, self).__init__(map, seed)

        self.gamma = 1  # v衰减系数
        self.restitution = 0.5
        self.print_log = False
        self.print_log2 = False
        self.tau = 0.1

        self.speed_cap =  100

        self.draw_obs = True
        self.show_traj = True

        #self.is_render = True

    def check_overlap(self):
        #todo
        pass

    def get_reward(self):

        agent_reward = [0. for _ in range(self.agent_num)]


        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                agent_reward[agent_idx] = 100.

        return agent_reward

    def is_terminal(self):

        if self.step_cnt >= self.max_step:
            return True

        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True

        return False





    def step(self, actions_list):

        previous_pos = self.agent_pos

        time1 = time.time()
        self.stepPhysics(actions_list, self.step_cnt)
        time2 = time.time()
        #print('stepPhysics time = ', time2 - time1)
        self.speed_limit()

        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        step_reward = self.get_reward()
        done = self.is_terminal()

        time3 = time.time()
        obs_next = self.get_obs()
        time4 = time.time()
        #print('render time = ', time4-time3)
        # obs_next = 1
        #self.check_overlap()
        self.change_inner_state()

        return obs_next, step_reward, done, ''

