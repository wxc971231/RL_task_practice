import pygame

# color 宏
COLORS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'grey':  [176,196,222],
    'purple': [160, 32, 240],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'light green': [204, 255, 229],
    'sky blue': [0,191,255]
}

COLOR_TO_IDX = {
    'red': 7,
    'green': 1,
    'sky blue': 2,
    'yellow': 3,
    'grey': 4,
    'purple': 5,
    'black': 6,
    'light green': 0,

}

IDX_TO_COLOR = {
    0: 'light green',
    1: 'green',
    2: 'sky blue',
    3: 'yellow',
    4: 'grey',
    5: 'purple',
    6: 'black',
    7: 'red'
}

grid_node_width = 2     #for view drawing
grid_node_height = 2


class Viewer():
    def __init__(self, setting):
        pygame.init()
        width = setting["width"]
        height = setting["height"]
        edge = setting["edge"]
        self.WIN_SIZE = width+2*edge, height+2*edge
        # self.background = pygame.display.set_mode(self.WIN_SIZE)
        #
        # self.draw_background()
        self.color_list = [ [255, 0, 0], [0, 255, 0], [0,0,255]  , [0,0,0], [160, 32, 240]]
        # WIN_SIZE = 1000, 1000
    def set_mode(self):
        self.background = pygame.display.set_mode(self.WIN_SIZE)

    def draw_background(self):
        self.background.fill((255, 255, 255))

    def draw_ball(self, pos_list, agent_list):
        # self.background.fill((255, 255, 255))
        assert len(pos_list) == len(agent_list)
        for i in range(len(pos_list)):
            t = pos_list[i]
            r = agent_list[i].r
            color = agent_list[i].color
            #print('color in viewer', color)
            pygame.draw.circle(self.background, COLORS[color], t, r, 0)
            pygame.draw.circle(self.background, COLORS['black'], t, 2, 2)


    def draw_direction(self, pos_list, a_list):
        """
        :param pos_list: position of circle center
        :param a_list: acceleration of circle
        :return:
        """
        assert len(pos_list) == len(a_list)
        for i in range(len(pos_list)):
            a_x, a_y = a_list[i]
            if a_x != 0 or a_y != 0:
                t = pos_list[i]
                start_x, start_y = t
                end_x = start_x + a_x/5
                end_y = start_y + a_y/5

                pygame.draw.line(self.background, color = [0,0,0], start_pos=[start_x, start_y], end_pos = [end_x, end_y], width = 2)


    def draw_map(self, object):
        # (left, top), width, height
        #pygame.draw.rect(self.background, [0, 0, 0], [0, 200, 800, 400], 2) # black
        #pygame.draw.rect(self.background, [255, 0, 0], [700, 200, 2, 400], 1) # red
        #print("check color: ", object.color)
        if object.type == 'arc':
            pygame.draw.arc(self.background, COLORS[object.color], object.init_pos, object.start_radian, object.end_radian, object.width)
            # pygame.draw.arc(self.background, COLORS[object.color], object.init_pos, object.start_radian-0.02, object.end_radian+0.02, object.width)

        else:
            s, e = object.init_pos
            pygame.draw.line(surface = self.background, color = COLORS[object.color], start_pos = s, end_pos = e, width = object.width)


    def draw_trajectory(self, trajectory_list, agent_list):
        for i in range(len(trajectory_list)):
            for t in trajectory_list[i]:
                pygame.draw.circle(self.background, COLORS[agent_list[i].color], t, 2, 1)

    def draw_obs(self, points, agent_list):
        for b in range(len(points)):
            pygame.draw.lines(self.background, agent_list[b].color, 1, points[b], 2)

    def draw_energy_bar(self, agent_list):
        coord = [570 + 70 * i for i in range(len(agent_list))]
        for agent_idx in range(len(agent_list)):
            remaining_energy = agent_list[agent_idx].energy/agent_list[agent_idx].energy_cap
            start_pos = [coord[agent_idx], 100]
            end_pos=  [coord[agent_idx] + 50*remaining_energy, 100]
            pygame.draw.line(self.background, color=COLORS[agent_list[agent_idx].color], start_pos=start_pos,
                             end_pos=end_pos, width = 5)




    def draw_view(self, obs, agent_list):       #obs: [2, 100, 100] list

        #draw agent 1, [50, 50], [50+width, 50], [50, 50+height], [50+width, 50+height]
        coord = [580 + 70 * i for i in range(len(obs))]
        for agent_idx in range(len(obs)):
            matrix = obs[agent_idx]
            obs_weight, obs_height = matrix.shape[0], matrix.shape[1]
            y = 30 - obs_height
            for row in matrix:
                x = coord[agent_idx]- obs_height/2
                for item in row:
                    pygame.draw.rect(self.background, COLORS[IDX_TO_COLOR[int(item)]], [x,y,grid_node_width, grid_node_height])
                    x+= grid_node_width
                y += grid_node_height

            pygame.draw.circle(self.background, COLORS[agent_list[agent_idx].color], [coord[agent_idx]+10, 55 + agent_list[agent_idx].r],
                               agent_list[agent_idx].r, width=0)
            pygame.draw.circle(self.background, COLORS["black"], [coord[agent_idx]+10, 55 + agent_list[agent_idx].r], 2,
                               width=0)

            pygame.draw.lines(self.background, points =[[566+70*agent_idx,5],[566+70*agent_idx, 55], [566+50+70*agent_idx, 55], [566+50+70*agent_idx, 5]], closed=True,
                              color = COLORS[agent_list[agent_idx].color], width=2)


pygame.init()
font = pygame.font.Font(None, 18)
def debug(info, y = 10, x=10):
    display_surf = pygame.display.get_surface()
    debug_surf = font.render(str(info), True, (0,0,0))
    debug_rect = debug_surf.get_rect(topleft = (x,y))
    display_surf.blit(debug_surf, debug_rect)

