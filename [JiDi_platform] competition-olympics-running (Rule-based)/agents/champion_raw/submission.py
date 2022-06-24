import imp
from importlib.resources import path
import math
import random
import time
from queue import Queue
import cv2
import numpy as np
import json
import os

def discrete_radius(radius):
    radius_list = [9.375, 12.5, 15, 18.75, 20]
    return sorted(radius_list, key=lambda x: (x - radius * 6) ** 2)[0] / 6


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def direction(p1, p2):
    h_dis = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    if h_dis < 1e-10:
        return 0.
    h_angle = math.acos((p2[0] - p1[0]) / h_dis) * 180 / math.pi
    return 180 - h_angle if p2[1] > p1[1] else -(180 - h_angle)


init_angle = 30.
zero_angle = -180.
scaled_size = 50
scale_level = 2
view_size = 12.5
map_size = 2000

direction_dim = 19
pos_dim = 3
target_num = 3
history_num = 3
target_piece = 10

dx = [-1, 0, 1, 0, -1, -1, 1, 1]
dy = [0, -1, 0, 1, -1, 1, -1, 1]


def get_position(img1, img2, position):
    diff = 1e10
    pos = None

    r = 9

    for ii in range(r):
        for jj in range(r):
            nx = int(ii + position[0] - r // 2)
            ny = int(jj + position[1] - r // 2)
            if nx > 0:
                if ny > 0:
                    image1 = img1[:-nx, :-ny]
                    image2 = img2[nx:, ny:]
                elif ny < 0:
                    image1 = img1[:-nx, -ny:]
                    image2 = img2[nx:, :ny]
                else:
                    image1 = img1[:-nx, :]
                    image2 = img2[nx:, :]
            elif nx < 0:
                if ny > 0:
                    image1 = img1[-nx:, :-ny]
                    image2 = img2[:nx, ny:]
                elif ny < 0:
                    image1 = img1[-nx:, -ny:]
                    image2 = img2[:nx, :ny]
                else:
                    image1 = img1[-nx:, :]
                    image2 = img2[:nx, :]
            else:
                if ny > 0:
                    image1 = img1[:, :-ny]
                    image2 = img2[:, ny:]
                elif ny < 0:
                    image1 = img1[:, -ny:]
                    image2 = img2[:, :ny]
                else:
                    image1 = img1
                    image2 = img2

            diff_sum = np.sum((image1 - image2) ** 2, where=np.logical_and(np.logical_and(image1 >= 0, image1 < 150),
                                                                           np.logical_and(image2 >= 0, image2 < 150)))

            if diff > diff_sum:
                pos = (nx, ny)
                diff = diff_sum

    return pos

class RuleAgent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.radius = None
        self.past = None
        self.game_map = dict()
        self.game_map["objects"] = list()
        self.game_map["agents"] = list()
        self.game_map["view"] = {
            "width": 600,
            "height": 600,
            "edge": 50
        }
        self.server = None
        self.last_action = None
        self.v = 0.
        self.x = 0.
        self.y = 0.
        self.angle = zero_angle
        self.t1 = self.t2 = self.t3 = self.t4 = self.t5 = self.t6 = self.t7 = 0
        self.end_flag = False
        self.arrows = []
        self.points = None
        self.force = 0
        self.history = []
        self.target_history = []

        self.global_map = np.ones([map_size, map_size, 4], dtype=np.float32) * 150.
        # self.seed(seed)

    def reset(self):
        self.radius = None
        self.past = None
        self.game_map = dict()
        self.game_map["objects"] = list()
        self.game_map["agents"] = list()
        self.game_map["view"] = {
            "width": 600,
            "height": 600,
            "edge": 50
        }
        self.server = None
        self.last_action = None
        self.v = 0.
        self.x = 0.
        self.y = 0.
        self.angle = zero_angle
        self.force = 0
        self.end_flag = False
        self.points = None
        self.arrows = []
        self.history = []
        self.target_history = []
        self.global_map = np.ones([map_size, map_size, 4], dtype=np.float32) * 150.

    def seed(self, seed=None):
        random.seed(seed)

    def process_obs(self, obs):

        processed_obs = np.zeros(obs.shape + (4,), dtype=np.float32)

        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i, j] == 1 or obs[i, j] == 5:
                    pass  # processed_obs[i, j, 3] = 30
                elif obs[i, j] == 4:
                    processed_obs[i, j, 0] = 1.
                    processed_obs[i, j, 3] = 60
                elif obs[i, j] == 6:
                    processed_obs[i, j, 1] = 1.
                    processed_obs[i, j, 3] = 120
                elif obs[i, j] == 7:
                    processed_obs[i, j, 2] = 1.
                    processed_obs[i, j, 3] = 90
                    self.end_flag = True
        return processed_obs

    def rotate(self, img, angle):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        obs_warped = np.zeros(img.shape, dtype=np.float32)
        for i in range(img.shape[2]):
            obs_warped[:, :, i] = cv2.warpAffine(img[:, :, i], M, (cols, rows), borderValue=-0.00001)
        return obs_warped

    def update_map(self, img, pos_x, pos_y):
        x = int(pos_x + 0.5)
        y = int(pos_y + 0.5)
        for i in range(scaled_size):
            for j in range(scaled_size):
                if img[i, j, 3] >= 0 and self.global_map[
                    x + i - scaled_size // 2 + map_size // 2, y + j - scaled_size // 2 + map_size // 2, 3] >= 150:
                    self.global_map[
                        x + i - scaled_size // 2 + map_size // 2, y + j - scaled_size // 2 + map_size // 2] = img[i, j]

    def update_self_arround(self, r_scale=1.2):
        r = self.radius * scale_level * r_scale
        for i in range(int(r * 2 + 1)):
            for j in range(int(r * 2 + 1)):
                x = int(self.x + i - r) + map_size // 2
                y = int(self.y + j - r) + map_size // 2
                if distance((int(self.x + i - r), int(self.y + j - r)), (self.x, self.y)) <= r and self.global_map[
                    x, y, 3] >= 150:
                    self.global_map[x, y] = 0

    def add_arrows(self, img, now_x, now_y):
        arrow_map = np.zeros_like(img, dtype=int)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > 0.9 and arrow_map[i, j] == 0:
                    x1, x2, y1, y2 = i, i + 1, j, j + 1
                    q = Queue()
                    q.put((i, j))
                    while not q.empty():
                        x, y = q.get()
                        if x1 > x:
                            x1 = x
                        if x2 < x + 1:
                            x2 = x + 1
                        if y1 > y:
                            y1 = y
                        if y2 < y + 1:
                            y2 = y + 1
                        for k in range(4):
                            nx = dx[k] + x
                            ny = dy[k] + y
                            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and img[nx, ny] > 0.9 and arrow_map[
                                nx, ny] == 0:
                                arrow_map[nx, ny] = 1
                                q.put((nx, ny))
                    arrow_direction = None
                    n_img = arrow_map[x1:x2, y1:y2]
                    e_list, w_list, n_list, s_list = [], [], [], []
                    e = 0
                    for k in range(x1, x2):
                        my1 = y2
                        my2 = y1 - 1
                        b = 0
                        c = 0
                        d = 0
                        for ll in range(y1, y2):
                            if arrow_map[k, ll] == 1:
                                b = 1
                                if my1 > ll:
                                    my1 = ll
                                if my2 < ll:
                                    my2 = ll
                            if arrow_map[k, ll] == 0 and b > 0:
                                c = 1
                            if arrow_map[k, ll] == 1 and c > 0:
                                d = 1
                        e_list.append(my1)
                        w_list.append(my2)
                        if d > 0:
                            e = 1
                    if e == 0:
                        if e_list[0] < max(e_list) and e_list[-1] < max(e_list):
                            arrow_direction = 'E'
                            pos = (e_list.index(max(e_list)) + x1 - img.shape[0] // 2 + now_x,
                                   max(e_list) - img.shape[1] // 2 + now_y)
                        if w_list[0] > min(w_list) and w_list[-1] > min(w_list):
                            arrow_direction = 'W'
                            pos = (w_list.index(min(w_list)) + x1 - img.shape[0] // 2 + now_x,
                                   min(w_list) - img.shape[1] // 2 + now_y)

                    e = 0
                    for k in range(y1, y2):
                        mx1 = x2
                        mx2 = x1 - 1
                        b = 0
                        c = 0
                        d = 0
                        for ll in range(x1, x2):
                            if arrow_map[ll, k] == 1:
                                b = 1
                                if mx1 > ll:
                                    mx1 = ll
                                if mx2 < ll:
                                    mx2 = ll
                            if arrow_map[ll, k] == 0 and b > 0:
                                c = 1
                            if arrow_map[ll, k] == 1 and c > 0:
                                d = 1
                        s_list.append(mx1)
                        n_list.append(mx2)
                        if d > 0:
                            e = 1
                    if e == 0:
                        if s_list[0] < max(s_list) and s_list[-1] < max(s_list):
                            arrow_direction = 'S'
                            pos = (max(s_list) - img.shape[0] // 2 + now_x,
                                   s_list.index(max(s_list)) + y1 - img.shape[1] // 2 + now_y)
                        if n_list[0] > min(n_list) and n_list[-1] > min(n_list):
                            arrow_direction = 'N'
                            pos = (min(n_list) - img.shape[0] // 2 + now_x,
                                   n_list.index(min(n_list)) + y1 - img.shape[1] // 2 + now_y)

                    if arrow_direction is not None:
                        redundant = False
                        for arrow in self.arrows:
                            if distance(arrow[0], pos) < 10:
                                redundant = True
                        if not redundant:
                            self.arrows.append([pos, arrow_direction, True])

                    for k in range(len(self.arrows)):
                        for ll in range(len(self.arrows)):
                            if k != ll and distance(self.arrows[k][0], self.arrows[ll][0]) < 60 and self.in_direction(
                                    self.arrows[ll], self.arrows[k][0]) > 0:
                                self.arrows[ll][2] = False

    def route(self, edge):
        # img = cv2.resize(self.global_map, (map_size // 2, map_size // 2))
        img = self.global_map

        end_map = np.zeros(img.shape[:2], dtype=int)
        past_x = np.ones(img.shape[:2], dtype=int) * -1
        past_y = np.ones(img.shape[:2], dtype=int) * -1

        for ep in edge['points']:
            end_map[ep[0], ep[1]] = 2

        # x = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius*2) * scale_level
        # y = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius*2) * scale_level
        x = self.x
        y = self.y
        x1, y1 = int(x) + map_size // 2, int(y) + map_size // 2
        q = Queue()
        q.put((x1, y1))
        end_map[x1, y1] = 1
        while not q.empty():
            x, y = q.get()
            for k in range(4):
                nx = dx[k] * 3 + x
                ny = dy[k] * 3 + y
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    if end_map[nx, ny] == 2:
                        tx, ty = x, y
                        point_list = [(x, y)]
                        while tx != x1 or ty != y1:
                            tx, ty = past_x[tx, ty], past_y[tx, ty]
                            point_list.append((tx, ty))
                        return point_list
                    elif end_map[nx, ny] == 0 and img[nx, ny, 1] <= 0:
                        end_map[nx, ny] = 1
                        q.put((nx, ny))
                        past_x[nx, ny] = x
                        past_y[nx, ny] = y
        raise 1

    def to_ends(self):
        if not self.end_flag:
            return None

        # img = cv2.resize(self.global_map, (map_size // 2, map_size // 2))
        img = self.global_map

        end_map = np.zeros(img.shape[:2], dtype=int)
        past_x = np.ones(img.shape[:2], dtype=int) * -1
        past_y = np.ones(img.shape[:2], dtype=int) * -1

        # x = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius*2) * scale_level
        # y = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius*2) * scale_level
        x = self.x
        y = self.y
        x1, y1 = int(x) + map_size // 2, int(y) + map_size // 2
        q = Queue()
        q.put((x1, y1))
        end_map[x1, y1] = 1
        while not q.empty():
            x, y = q.get()
            end_map[x, y] = 1
            if img[x, y, 2] > 0.1:
                tx, ty = x, y
                point_list = [(x, y)]
                while tx != x1 or ty != y1:
                    tx, ty = past_x[tx, ty], past_y[tx, ty]
                    point_list.append((tx, ty))
                return point_list
            for k in range(4):
                nx = dx[k] * 3 + x
                ny = dy[k] * 3 + y
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and img[nx, ny, 1] <= 0 and end_map[nx, ny] == 0:
                    end_map[nx, ny] = 1
                    q.put((nx, ny))
                    past_x[nx, ny] = x
                    past_y[nx, ny] = y
        return None

    def get_angle_score(self, point):
        angle = direction(self.curr_pos(), point)
        da = angle - self.angle
        if da > 180:
            da -= 360
        if da < -180:
            da += 360
        return -abs(da) / 180 * 0.5

    def get_distance_score(self, point):
        d = distance(self.curr_pos(), point)

        return -d / 50 if d > 70 else 0.

    def get_edges(self):

        t = time.time()
        # points = self.to_ends()
        # self.t5 += time.time() - t
        # if points is not None:
        #     return points

        # img = cv2.resize(self.global_map, (map_size // 2, map_size // 2))
        img = self.global_map

        edge_map = np.zeros(img.shape[:2], dtype=int)
        past_x = np.ones(img.shape[:2], dtype=int) * -1
        past_y = np.ones(img.shape[:2], dtype=int) * -1

        # x = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius*2) * scale_level
        # y = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius*2) * scale_level
        x = self.x
        y = self.y
        start_x, start_y = int(x) + map_size // 2, int(y) + map_size // 2
        q = Queue()
        q.put((start_x, start_y))
        edge_map[start_x, start_y] = 1
        edge_point_list = []
        while not q.empty():
            x, y = q.get()
            if img[x, y, 2] > 0.1:
                tx, ty = x, y
                point_list = [(x, y)]
                while tx != start_x or ty != start_y:
                    tx, ty = past_x[tx, ty], past_y[tx, ty]
                    point_list.append((tx, ty))
                return point_list
            for k in range(4):
                nx = dx[k] * 3 + x
                ny = dy[k] * 3 + y
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and edge_map[nx, ny] == 0:
                    if img[nx, ny, 1] <= 0:
                        edge_map[nx, ny] = 1
                        q.put((nx, ny))
                        past_x[nx, ny] = x
                        past_y[nx, ny] = y
                    elif img[nx, ny, 1] >= 150:
                        edge_map[nx, ny] = 2
                        edge_point_list.append((nx, ny))
                        past_x[nx, ny] = x
                        past_y[nx, ny] = y

        edge_list = []
        for ep in edge_point_list:
            if edge_map[ep[0], ep[1]] == 2:
                center_x, center_y = 0, 0
                edge = {'points': []}
                x1, y1 = ep[0], ep[1]
                q = Queue()
                q.put((x1, y1))
                edge_map[x1, y1] = 3
                while not q.empty():
                    x, y = q.get()
                    edge['points'].append((x, y))
                    center_x += x
                    center_y += y
                    for k in range(8):
                        nx = dx[k] * 3 + x
                        ny = dy[k] * 3 + y
                        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and edge_map[nx, ny] == 2 and len(
                                edge['points']) < 20:
                            edge_map[nx, ny] = 3
                            q.put((nx, ny))
                if len(edge['points']) > 4:
                    edge['center'] = (center_x / len(edge['points']), center_y / len(edge['points']))
                    edge_list.append(edge)

        if len(edge_list) == 0:
            return None

        reward = -100000
        e = None

        for edge in edge_list:
            num = 0
            dist = 1e10
            for arrow in self.arrows:
                d = distance(edge['center'], arrow[0])
                flag = self.in_direction(arrow, edge['center'])
                if arrow[2] and d < 70 and flag > 0:
                    num += 1
                    if dist > d + 1:
                        dist = d + 1
                # elif not arrow[2] and d < 30:
                #     num -= 10

            num += self.get_angle_score(edge['center'])
            num += self.get_distance_score(edge['center'])

            v = num if num < 0 else num / dist
            if reward < v:
                reward = v
                e = edge

        point = None
        dis = 1e10

        for p in e['points']:
            d = distance(p, e['center'])
            if dis > d:
                dis = d
                point = p

        tx, ty = point[0], point[1]
        point_list = [(point[0], point[1])]
        while tx != start_x or ty != start_y:
            tx, ty = past_x[tx, ty], past_y[tx, ty]
            point_list.append((tx, ty))
        return point_list

    def in_direction(self, arrow, point):
        if distance(arrow[0], point) < 10:
            return 1
        dis_x = point[0] - arrow[0][0]
        dis_y = point[1] - arrow[0][1]

        if arrow[1] == 'E':
            if dis_y > 0 and abs(dis_x) < abs(dis_y) * 5:
                return 1
            if dis_y < 0 and abs(dis_x) < abs(dis_y) * 5:
                return -10
        elif arrow[1] == 'S':
            if dis_x > 0 and abs(dis_y) < abs(dis_x) * 5:
                return 1
            if dis_x < 0 and abs(dis_y) < abs(dis_x) * 5:
                return -10
        elif arrow[1] == 'W':
            if dis_y < 0 and abs(dis_x) < abs(dis_y) * 5:
                return 1
            if dis_y > 0 and abs(dis_x) < abs(dis_y) * 5:
                return -10
        elif arrow[1] == 'N':
            if dis_x < 0 and abs(dis_y) < abs(dis_x) * 5:
                return 1
            if dis_x > 0 and abs(dis_y) < abs(dis_x) * 5:
                return -10

        return 0

    def get_direction(self):
        t = time.time()
        points = self.to_ends()
        self.t5 += time.time() - t
        if points is not None:
            return points
        t = time.time()
        edges = self.get_edges()

        if len(edges) == 0:
            return None

        reward = -100000
        e = None

        for edge in edges:
            num = 0
            dist = 1e10
            for arrow in self.arrows:
                d = distance(edge['center'], arrow[0])
                if d < 70:
                    flag = self.in_direction(arrow, edge['center'])
                    if num < 0 or arrow[2]:
                        num += flag
                        if flag > 0 and dist > d:
                            dist = d

            v = num if num < 0 else num / dist
            if reward < v:
                reward = v
                e = edge
        self.t6 += time.time() - t

        t = time.time()
        r = self.route(e)
        self.t7 += time.time() - t
        return r

    def fix_action(self, force, angle):
        if len(self.history) > 3:
            a = direction(self.history[-4], self.curr_pos()) - self.angle
            if a > 180:
                a -= 360
            elif a < -180:
                a += 360
            # print(a, self.v)
            # if abs(a) < 10 and self.v >= 3.3:
            #     force = force/2

            # a = direction(self.history[-4], (self.x + map_size // 2, self.y + map_size // 2)) - angle
            # if a > 180:
            #     a -= 360
            # elif a < -180:
            #     a += 360

            if abs(angle) < 29 and abs(a) > 40 and self.v > 2.5:
                new_angle = angle - a * min(abs(a), 20) / 8
                if new_angle > 30:
                    new_angle = 30
                elif new_angle < -30:
                    new_angle = -30
                # print(new_angle)
                angle = new_angle
        return force, angle

    def get_angle_from_curr(self, point):

        angle = direction(self.curr_pos(), point) - self.angle
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        if angle > 30:
            angle = 30
        elif angle < -30:
            angle = -30
        return angle

    def get_action(self, points):
        if self.v <= 2.01:
            power = 200
        else:
            power = 200 / self.v * 2
        # point = points[len(points) // 2]
        pos = self.curr_pos()
        for p in points[:-1]:
            d = int(distance(p, pos))
            flag = True
            for i in range(d):
                x = int((p[0] - pos[0]) / d * i + pos[0])
                y = int((p[1] - pos[1]) / d * i + pos[1])
                if self.global_map[x, y, 1] > 0:
                    flag = False
            if flag:
                return power, self.get_angle_from_curr(p)

        return power, self.get_angle_from_curr(p)
        # angle = self.get_angle_from_curr(point)
        #
        # return 75, angle

    def act(self, obs):
        obs = self.process_obs(obs)
        n_obs = cv2.resize(obs[:, :], (scaled_size, scaled_size))
        # if len(self.history) > 1:
        #    print(distance(self.history[-2], self.history[-1]))

        if self.radius is None:
            force = 0
            angle = init_angle
            # self.radius = -1
            obs_warped = self.rotate(n_obs, zero_angle)
            self.radius = 2.5
            self.update_map(obs_warped, (self.radius + view_size) * scale_level, 0)
            self.update_self_arround(1.5)
            nx, ny = (self.radius + view_size) * scale_level, 0
            self.add_arrows(obs_warped[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
            points = self.get_edges()
            if points is None:
                force, angle = self.force, 0
            else:
                force, angle = self.get_action(points)
                self.points = points
        else:
            t = time.time()
            obs_warped = self.rotate(n_obs, -self.angle)
            self.t1 += time.time() - t
            if self.radius < 0:
                # matched offset center of view
                pos = get_position(obs_warped[:, :, 3], self.past[:, :, 3], np.array([-4, -14], dtype=np.float32))
                dis = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                self.radius = discrete_radius(dis / 0.5 / 2 * math.sin(math.pi * 5 / 12) - view_size)
                # self.game_map['agents'].append(
                #     Agent(
                #         mass=16,
                #         r=self.radius,
                #         position=[0., 0.],
                #         color="green"
                #     ))
                # self.server = FakeEnv(self.game_map)
                # self.radius = 0
                tmp = cv2.resize(self.global_map, (400, 400))
                self.update_map(self.past, (self.radius + view_size) * scale_level, 0)
                nx, ny = (self.radius + view_size) * scale_level, 0
                self.add_arrows(self.past[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
                tmp = cv2.resize(self.global_map, (400, 400))
                nx, ny = (self.radius + view_size) * scale_level + pos[0], pos[1]
                self.update_map(obs_warped, nx, ny)
                # self.add_arrows(obs_warped[:, :, 0], int(nx+map_size//2), int(ny+map_size//2))
                tmp = cv2.resize(self.global_map, (400, 400))
            else:

                t = time.time()
                # predicted center of view
                x = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                y = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level

                # matched offset center of view
                pos = get_position(obs_warped[:, :, 3],
                                   self.global_map[int(x) - scaled_size // 2 + map_size // 2:int(
                                       x) + scaled_size // 2 + map_size // 2,
                                   int(y) - scaled_size // 2 + map_size // 2:int(
                                       y) + scaled_size // 2 + map_size // 2, 3],
                                   np.array([0, 0], dtype=np.float32))

                # matched center of view
                nx, ny = x + pos[0], y + pos[1]
                # print(pos)

                # fixed pos of agent
                self.x = nx + math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                self.y = ny + math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                # print(self.x, self.y, self.angle)

                self.t2 += time.time() - t
                t = time.time()
                self.update_map(obs_warped, nx, ny)
                self.t3 += time.time() - t

            t = time.time()
            self.update_self_arround()
            # tmp = cv2.resize(self.global_map[920:1080, 920:1080, 3], (50, 50))
            tmp = cv2.resize(self.global_map[:, :, 3], (400, 400))
            self.add_arrows(obs_warped[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
            points = self.get_edges()
            if points is None:
                force, angle = self.force, 0
                self.points = points
            else:
                force, angle = self.get_action(points)
            self.t4 += time.time() - t
            # force = random.uniform(self.force_range[0], self.force_range[1])
            # angle = random.uniform(self.angle_range[0], self.angle_range[1])
        self.past = obs_warped

        self.history.append(self.curr_pos())
        self.target_history.append(np.array(self.points))

        if len(self.history) > 3:
            self.v = distance(self.history[-4], self.history[-1]) / 3
        else:
            self.v = 1

        force, angle = self.fix_action(force, angle)
        self.last_action = [force, angle]
        self.force = force
        self.angle += angle
        if self.angle < -180:
            self.angle += 360.
        if self.angle > 180:
            self.angle -= 360.
        # print(self.angle)
        # print(self.v)

        # print(self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7)
        return [force, angle]

    def get_obstacle(self, angle):
        a = self.angle + angle
        for i in range(1, 100):
            x = int(self.x - math.cos(-a / 180 * math.pi) * i) + map_size // 2
            y = int(self.y - math.sin(-a / 180 * math.pi) * i) + map_size // 2
            if self.global_map[x, y, 1] >= 150:
                return [0, math.log(i)]
            if self.global_map[x, y, 1] > 0.1:
                return [math.log(i), 0]
        return [0, 0]

    def curr_pos(self):
        return int(self.x) + map_size // 2, int(self.y) + map_size // 2

    def get_target(self, i):
        if self.points is None or i >= len(self.points):
            return [-1, -1, -1]
        angle = self.get_angle_from_curr(self.points[-i])
        return [math.cos(angle), math.sin(angle), math.log(distance(self.curr_pos(), self.points[-i]) + 1.)]

    def get_history(self, i):
        if self.history is None or i >= len(self.history):
            return [-1, -1, -1]
        angle = self.get_angle_from_curr(self.history[-i])
        return [math.cos(angle), math.sin(angle), math.log(distance(self.curr_pos(), self.history[-i]) + 1.)]

    def get_reward(self):
        reward = 0
        for i in range(1, 1 + 1):
            if len(self.target_history) > i * target_piece and self.target_history[-i * target_piece].shape[
                0] > i * target_piece:
                reward -= distance(self.curr_pos(), self.target_history[-i * target_piece][-i * target_piece]) / 10000
                print('d', distance(self.curr_pos(), self.target_history[-i * target_piece][-i * target_piece]),
                      self.curr_pos(), self.target_history[-i * target_piece][-i * target_piece])
        print(reward)
        return reward

    def generate_state(self):

        # obs = np.zeros([direction_dim*2+3*3+10*3], dtype=float)
        obs = []
        for i in range(direction_dim):
            obs.extend(self.get_obstacle(i * 10 - 90))

        for i in range(1, target_num + 1):
            obs.extend(self.get_target(i * target_piece))

        for i in range(history_num):
            obs.extend(self.get_history(i + 1))

        return np.array(obs, dtype=float)

    def get_states(self, obs):
        obs = self.process_obs(obs)
        n_obs = cv2.resize(obs[:, :], (scaled_size, scaled_size))
        if len(self.history) > 1:
            print(distance(self.history[-2], self.history[-1]))

        if self.radius is None:
            force = 0
            angle = init_angle
            # self.radius = -1
            obs_warped = self.rotate(n_obs, zero_angle)
            self.radius = 2.5
            self.update_map(self.past, (self.radius + view_size) * scale_level, 0)
            self.update_self_arround(1.8)
            nx, ny = (self.radius + view_size) * scale_level, 0
            self.add_arrows(obs_warped[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
            points = self.get_edges()
            if points is None:
                force, angle = self.force, 0
            else:
                force, angle = self.get_action(points)
                self.points = points
        else:
            t = time.time()
            obs_warped = self.rotate(n_obs, -self.angle)
            self.t1 += time.time() - t
            if self.radius < 0:
                # matched offset center of view
                pos = get_position(obs_warped[:, :, 3], self.past[:, :, 3], np.array([-4, -14], dtype=np.float32))
                dis = math.sqrt(pos[0] ** 2 + pos[1] ** 2)
                self.radius = discrete_radius(dis / 0.5 / 2 * math.sin(math.pi * 5 / 12) - view_size)
                # self.game_map['agents'].append(
                #     Agent(
                #         mass=16,
                #         r=self.radius,
                #         position=[0., 0.],
                #         color="green"
                #     ))
                # self.server = FakeEnv(self.game_map)
                # self.radius = 0
                tmp = cv2.resize(self.global_map, (400, 400))
                self.update_map(self.past, (self.radius + view_size) * scale_level, 0)
                nx, ny = (self.radius + view_size) * scale_level, 0
                self.add_arrows(self.past[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
                tmp = cv2.resize(self.global_map, (400, 400))
                nx, ny = (self.radius + view_size) * scale_level + pos[0], pos[1]
                self.update_map(obs_warped, nx, ny)
                # self.add_arrows(obs_warped[:, :, 0], int(nx+map_size//2), int(ny+map_size//2))
                tmp = cv2.resize(self.global_map, (400, 400))
                self.update_self_arround(1.8)
            else:

                t = time.time()
                # predicted center of view
                x = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                y = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level

                # matched offset center of view
                pos = get_position(obs_warped[:, :, 3],
                                   self.global_map[int(x) - scaled_size // 2 + map_size // 2:int(
                                       x) + scaled_size // 2 + map_size // 2,
                                   int(y) - scaled_size // 2 + map_size // 2:int(
                                       y) + scaled_size // 2 + map_size // 2, 3],
                                   np.array([0, 0], dtype=np.float32))

                # matched center of view
                nx, ny = x + pos[0], y + pos[1]
                # print(pos)

                # fixed pos of agent
                self.x = nx + math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                self.y = ny + math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
                #print(self.curr_pos(), self.angle)

                self.t2 += time.time() - t
                t = time.time()
                self.update_map(obs_warped, nx, ny)
                self.t3 += time.time() - t

            t = time.time()
            self.update_self_arround()
            # tmp = cv2.resize(self.global_map[920:1080, 920:1080, 3], (50, 50))
            tmp = cv2.resize(self.global_map[:, :, 3], (400, 400))
            self.add_arrows(obs_warped[:, :, 0], int(nx + map_size // 2), int(ny + map_size // 2))
            points = self.get_edges()
            if points is None:
                force, angle = self.force, 0
            else:
                force, angle = self.get_action(points)
                self.points = points
            self.t4 += time.time() - t
            # force = random.uniform(self.force_range[0], self.force_range[1])
            # angle = random.uniform(self.angle_range[0], self.angle_range[1])

        self.past = obs_warped
        # self.last_action = [force, angle]
        # self.force = force
        # self.angle += angle
        # if self.angle < -180:
        #     self.angle += 360.
        # if self.angle > 180:
        #     self.angle -= 360.

        self.history.append(self.curr_pos())
        self.target_history.append(np.array(self.points))

        # print(self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7)
        state, reward = self.generate_state(), self.get_reward()
        return state, reward

    def update_action(self, force, angle):

        self.last_action = [force, angle]
        self.force = force
        self.angle += angle
        if self.angle < -180:
            self.angle += 360.
        if self.angle > 180:
            self.angle -= 360.

    def choose_action(self,observation) -> None:
        actions = self.act(observation['obs'])
        wrapped_actions = [[actions[0]], [actions[1]]]
        return wrapped_actions

agent = RuleAgent()
agent.reset()


def my_controller(observation, action_space, is_act_continuous=False):
    #actions = agent.act(observation)
    actions = agent.act(observation['obs'])
    wrapped_actions = [[actions[0]], [actions[1]]]
    return wrapped_actions
