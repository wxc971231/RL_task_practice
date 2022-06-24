from importlib.resources import path
import math
import random
from queue import Queue
from typing import Tuple
import cv2
import numpy as np
import random 
from PIL import Image

zero_angle = 180.   # 原始观测角度转 180 度为零角度
scaled_size = 50    # 超分后的地图尺寸 (50,50)
scale_level = 2     # 超分倍数
view_size = 12.5    # 超分前观测框1/2边长像素（超分后 12.5*2*2 = 50）
map_size = 2000     # 拼接后完整地图的尺寸（实际有效区域只占这个 (2000,2000) 中的一部分，各个地图尺寸不同）
 
dx = [-1, 0, 1, 0, -1, -1, 1, 1]
dy = [0, -1, 0, 1, -1, 1, -1, 1]

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def to_absolute_coord(coord) -> float:
    """ Convert center coordinates to absolute coordinates
    
    :param coord: center coord that need to convert
    :return: absolute coordinates
    :rtype: float
    """

    return int(coord) + map_size // 2

def angel_standardized(angle) -> float:
    """ Standardize angle

    :param angle: raw angle
    :return: standardized angle, constrainted to [-180,180]
    :rtype: float
    """

    angle = angle - 360 if angle > 180 else angle
    angle = angle + 360 if angle < -180 else angle
    return angle

def angel_output_constrain(angle) -> float:
    """ Constrain output angle

    :param angle: raw angle
    :return: aviliable angle, constrainted to [-30,30]
    :rtype: float
    """

    angle = 30 if angle > 30 else angle
    angle = -30 if angle < -30 else angle
    return angle

def direction(p1, p2) -> float:
    """ Get direction of the line connect from p1 to p2, i.e. the angle between the line and the negative X-axis (South) direction
    
    :param p1: the begin point of the line
    :param p2: the end point of the line
    :return: the direction of the line p1->p2
    :rtype: float
    """

    dis = distance(p1,p2)
    if dis < 1e-10:
        return 0.
    angle = math.acos((p2[0] - p1[0]) / dis) * 180 / math.pi    
    return 180 - angle if p2[1] > p1[1] else -(180 - angle)


def get_displacement(current_obs, last_obs, default_displacement) -> Tuple[int,int]:
    """ Get the displacement of agent motion within last game frame 

    :param current_obs: current wrapped observation of agent
    :param last_obs: wrapped observation of agent in the last frame, which is obtained from global_map basing on agent's position of last frame   
    :param default_displacement: default displacement
    :return: the displacement of agent motion within last game frame 
    :rtype: Tuple[int,int]

    :note: in order to get the displacement, calculate all the cases to match the current observation and last observation of agent
    """

    diff = 1e10
    pos = None
    r = 9

    for ii in range(r):
        for jj in range(r):
            # 以 default_displacement 为中心，遍历 r*r 区域，遍历待比较的偏移量
            nx = int(ii + default_displacement[0] - r // 2) 
            ny = int(jj + default_displacement[1] - r // 2)
            
            # 根据位移裁剪出比较区域
            if nx > 0:
                if ny > 0:
                    image1 = current_obs[:-nx, :-ny]
                    image2 = last_obs[nx:, ny:]
                elif ny < 0:
                    image1 = current_obs[:-nx, -ny:]
                    image2 = last_obs[nx:, :ny]
                else:
                    image1 = current_obs[:-nx, :]
                    image2 = last_obs[nx:, :]
            elif nx < 0:
                if ny > 0:
                    image1 = current_obs[-nx:, :-ny]
                    image2 = last_obs[:nx, ny:]
                elif ny < 0:
                    image1 = current_obs[-nx:, -ny:]
                    image2 = last_obs[:nx, :ny]
                else:
                    image1 = current_obs[-nx:, :]
                    image2 = last_obs[:nx, :]
            else:
                if ny > 0:
                    image1 = current_obs[:, :-ny]
                    image2 = last_obs[:, ny:]
                elif ny < 0:
                    image1 = current_obs[:, -ny:]
                    image2 = last_obs[:, :ny]
                else:
                    image1 = current_obs
                    image2 = last_obs

            # 计算比较区域的差距
            diff_sum = np.sum((image1 - image2) ** 2, where=np.logical_and(np.logical_and(image1 >= 0, image1 < 150),
                                                                           np.logical_and(image2 >= 0, image2 < 150)))
            # 找出差距最小的位移
            if diff > diff_sum:
                pos = (nx, ny)
                diff = diff_sum

    return pos

class RuleAgent:
    def __init__(self, seed=None):
        self.init = True                    # 初始化标志
        self.radius = 2.5                   # agent 示意圆半径，恒定为 2.5        
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.force = 0                      # 当前 agent 的出力
        self.angle = zero_angle             # 当前 agent 的绝对角度
        self.v = 0.                         # 当前 agent 速度
        self.x = 0.                         # 当前 agent 的中心坐标 x 坐标
        self.y = 0.                         # 当前 agent 的中心坐标 y 坐标
        self.arrows = []                    # 当前 agent 已经识别到的箭头列表，其中元素也是列表，结构为 [(x,y),'N'/'S'/'E'/'W',True/False]，第一项为顶点坐标，第二项为指向，第三项表示此箭头是否 “已经过”
        self.history = []                   # 由 agent 每一轮迭代中绝对坐标组成的列表
        self.global_map = np.ones([map_size, map_size, 4], dtype=np.float32) * 150. # 全局地图

        self.draw_edges = None              # 用于绘制当前已重建地图
        self.draw_path = None               # 用于绘制当前已重建地图
        # self.seed(seed)

    def reset(self):
        self.init = True
        self.radius = 2.5
        self.angle = zero_angle
        self.force = 0
        self.v = 0.
        self.x = 0.
        self.y = 0.
        self.arrows = []
        self.history = []
        self.global_map = np.ones([map_size, map_size, 4], dtype=np.float32) * 150.

        self.draw_edges = None              
        self.draw_path = None              

    def seed(self, seed=None):
        random.seed(seed)

    def curr_pos(self) -> Tuple[float,float]:
        """ Return the absolute coordinates of the current agent center position

        :return: absolute coordinates of the current agent center position
        :rtype: Tuple[float,float]
        """
        return to_absolute_coord(self.x), to_absolute_coord(self.y)

    def process_obs(self, obs) -> np.ndarray:
        """ Process the raw observations

        :param obs: raw observations, shape = (25,25)
        :return: processed observations
        :rtype: np.ndarray, shape = (25,25,4)

        :note: Added dimensions that each pixel associated with a four-dimensional vector, the first three of which form an one-hot vector to indicate the property of pixel, 
               and the value of last dimension is assigned according to the property. These dimensions will be used to identify arrow/wall/end signs in map
        """

        # 增加维度，processed_obs.shape = (25,25,4)
        # 每个像素对应一个四维向量，前三个维 one-hot 表示像素属性，根据属性不同对第四维赋值 60/90/120
        processed_obs = np.zeros(obs.shape + (4,), dtype=np.float32)

        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i, j] == 1 or obs[i, j] == 5:
                    pass  
                elif obs[i, j] == 4:                    # arrow
                    processed_obs[i, j, 0] = 1. 
                    processed_obs[i, j, 3] = 60
                elif obs[i, j] == 6:                    # wall
                    processed_obs[i, j, 1] = 1.
                    processed_obs[i, j, 3] = 120
                elif obs[i, j] == 7:                    # endline
                    processed_obs[i, j, 2] = 1.
                    processed_obs[i, j, 3] = 90

                    #np.save('map11.npy',self.global_map)
                    #self.draw_global_map()

        return processed_obs

    def rotate(self, img, angle) -> np.ndarray:
        """ Counterclockwise Rotate the image around the center of it

        :param img: Images that need to be rotated
        :param angle: Rotation Angle
        :return: wrapped observations
        :rtype: np.ndarray

        :note: Affine transformation is performed on first two dimensions
        """

        # 绕图像中心旋转在opencv实现时分成两步：1.绕左上角的原点旋转；2.平移使中心和原先的观测中心对齐。这两步可以组成一个仿射变换
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1) # 得到仿射变换矩阵（旋转中心在观测中心，顺时针旋转 agele 度，结果不缩放）
        obs_wrapped = np.zeros(img.shape, dtype=np.float32)      
        for i in range(img.shape[2]):   
            obs_wrapped[:, :, i] = cv2.warpAffine(img[:, :, i], M, (cols, rows), borderValue=-0.00001)   # 对每个维度执行仿射变换
        #obs_wrapped = cv2.warpAffine(img, M, (cols, rows), borderValue=-0.00001)
        return obs_wrapped

    def wrap_obs(self, obs) -> np.ndarray:
        """ Wrap the raw observations, the it carry more information and can be spliced into the global map

        :param obs: raw observations, shape = (25,25)
        :return: warpped observations
        :rtype: np.ndarray, shape = (50,50,4)
        """

        obs = self.process_obs(obs)                                   # 增加第三个维度 (25,25,4)
        obs = cv2.resize(obs[:, :], (scaled_size, scaled_size))       # 双线性插值放大到 (50,50,4)
        obs_wrapped = self.rotate(obs, -self.angle) 
        
        return obs_wrapped

    # img 是包装过的观测
    def update_map(self, wrap_obs, pos_x, pos_y) -> None:
        """ Splice current wrapped observation to update the global map 

        :param wrap_obs: wrapped observations, shape = (50,50,4)
        :param pos_x: the x center coord of current observation center
        :param pos_y: the y center coord of current observation center
        :return: None

        :note: The part that has been spliced won't be overwritten
        """

        # 观测中心的中心坐标，向上取整
        x = int(pos_x + 0.5)    
        y = int(pos_y + 0.5)

        # 遍历 wrap_obs 中所有像素，拼接到 self.global_map 上
        for i in range(scaled_size):
            for j in range(scaled_size):
                absolute_x = to_absolute_coord(x + i - scaled_size // 2) 
                absolute_y = to_absolute_coord(y + j - scaled_size // 2)
                #if wrap_obs[i, j, 3] >= 0 and self.global_map[absolute_x, absolute_y, 3] not in [0,60,90,120] :  # 这个条件是限制已经拼接过的部分不要被覆盖
                if wrap_obs[i, j, 3] >= 0 and self.global_map[absolute_x, absolute_y, 3] >= 150:
                    self.global_map[absolute_x, absolute_y] = wrap_obs[i, j]
                
    # 在 self.global_map 上，把当前观测中心一定半径内的地图清零（初始化是(150,150,150,150),清空后变成(0,0,0,0)）
    def update_self_arround(self, r_scale=1.2) -> None:
        """ Update the area near the agent in the global map

        :param r_scale: agent area zoom factor
        :return: None
        
        :note: The part that has been spliced won't be overwritten
        """

        r = self.radius * scale_level * r_scale
        for i in range(int(r * 2 + 1)):
            for j in range(int(r * 2 + 1)):
                # 中心坐标
                x_center = int(self.x + i - r)
                y_center = int(self.y + j - r)
                # 绝对坐标
                x_absolute = to_absolute_coord(x_center)
                y_absolute = to_absolute_coord(y_center)
                #if distance((x_center, y_center), (self.x, self.y)) <= r and self.global_map[x_absolute, y_absolute, 3] not in [0,60,90,120] :
                if distance((x_center, y_center), (self.x, self.y)) <= r and self.global_map[x_absolute, y_absolute, 3] >= 150 :
                    self.global_map[x_absolute, y_absolute] = 0
    
    # img 是原始 obs 经过 process_obs、超分和旋转后的第 0 维度，尺寸 (50,50)
    # process_obs后箭头位置 0 维度设为 1 了，但超分使箭头被模糊了，只有箭头中线位置附近的 0 维度 > 0.9 
    # now_x 和 now_y 是观测中心的绝对坐标 
    def add_arrows(self, img, now_x, now_y) -> None:
        """ Find and update arrow by current wrapped observation  
        
        :param img: The corresponding first two dimensions of current wrapped obeservation when the third dimension is 0, which is a indicator map of arrow sign, shape = (50,50)
        :param now_x: The x absolute coord of current agent center
        :param now_y: The y absolute coord of current agent center
        :return: None
        
        :note: Here we suppose the arrow can only point in one of the four directions: east, west, north and west. 
               The direction of arrow is defined in global map, which is rotated from raw observation, in such case "Left => West", "Right => East", "Up => North", "Down => South".
               Becaues the observation have been zoomed to twice its size, which will make it blurry, we believe (i,j) is an arrow pixel if img[i, j] > 0.9.
               New arrow will be added to self.arrows.
        """

        arrow_map = np.zeros_like(img, dtype=int)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                # 提取出箭头像素（超分后插值像素会模糊，img[i, j] > 0.9 即认为是靠近箭头线中心的像素）
                if img[i, j] > 0.9 and arrow_map[i, j] == 0:    
                    
                    # (i,j) 处是一个箭头像素，从此开始 bfs 找出该箭头范围，并把此箭头记录到 arrow_map 中 
                    xmin, xmax, ymin, ymax = i, i + 1, j, j + 1         
                    q = Queue()
                    q.put((i, j))
                    while not q.empty():
                        x, y = q.get()
                        xmin = x if xmin > x else xmin
                        xmax = x+1 if xmax < x+1 else xmax
                        ymin = y if ymin > y else ymin
                        ymax = y+1 if ymax < y+1 else ymax

                        # 在 img 尺寸范围内左上右下相邻像素 bfs，如果 img[nx, ny] > 0.9（有箭头）就用 arrow_map[nx, ny] = 1 记录
                        for k in range(4):
                            nx = dx[k] + x
                            ny = dy[k] + y
                            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and img[nx, ny] > 0.9 and arrow_map[nx, ny] == 0:
                                arrow_map[nx, ny] = 1
                                q.put((nx, ny))

                    # 分析上面得到的箭头范围内 arrow_map 的像素分布，判断此箭头指向
                    arrow_direction = None                         
                    n_img = arrow_map[xmin:xmax, ymin:ymax]         # 这是截取出的箭头（仅用于 debug 观察）
                    e_list, w_list, n_list, s_list = [], [], [], [] # 东西北南
                    dir_EW = True
                    
                    # 从南到北按行（x）扫描
                    for k in range(xmin, xmax):
                        arrow_ymin = ymax
                        arrow_ymax = ymin - 1
                        fst_find, fst_leave, sec_find = False,False,False 
 
                        for ll in range(ymin, ymax):
                            if arrow_map[k, ll] == 1:
                                fst_find = True                                     # 首次扫描到箭头像素
                                arrow_ymin = ll if arrow_ymin > ll else arrow_ymin  # arrow_ymin 记录箭头最靠东的 y 位置（y最小值）
                                arrow_ymax = ll if arrow_ymin < ll else arrow_ymax  # arrow_ymax 记录箭头最靠西的 y 位置（y最大值）
                            if arrow_map[k, ll] == 0 and fst_find:
                                fst_leave = True                                    # 找到后首次离开箭头记
                            if arrow_map[k, ll] == 1 and fst_leave:
                                sec_find = True                                     # 离开后又扫描到箭头像素

                        e_list.append(arrow_ymin)      
                        w_list.append(arrow_ymax)
                        # sec_find 代表（从南到北）按行（x）扫描过程中曾经截取到了箭头的两个尾巴，说明箭头方向和扫描方向 x 垂直
                        if sec_find > 0:            
                            dir_EW = 0

                    # （从南到北）按行（x）扫描的全过程中箭头都只出现一次，说明箭头只能是东或西方向
                    if dir_EW:
                        # 通过首尾坐标值和最大值判断出 > （E 方向），箭头顶点在 global_map 中的坐标记为 pos
                        if e_list[0] < max(e_list) and e_list[-1] < max(e_list):
                            arrow_direction = 'E'
                            pos = (e_list.index(max(e_list)) + xmin - img.shape[0] // 2 + now_x,
                                                max(e_list) - img.shape[1] // 2 + now_y)
                        # 通过首尾坐标值和最大值判断出 < （W 方向），箭头顶点在 global_map 中的坐标记为 pos
                        if w_list[0] > min(w_list) and w_list[-1] > min(w_list):
                            arrow_direction = 'W'
                            pos = (w_list.index(min(w_list)) + xmin - img.shape[0] // 2 + now_x,
                                                min(w_list) - img.shape[1] // 2 + now_y)

                    dir_SN = True
                    for k in range(ymin, ymax):
                        arrow_xmin = xmax
                        arrow_xmax = xmin - 1
                        fst_find, fst_leave, sec_find = False,False,False 

                        for ll in range(xmin, xmax):
                            if arrow_map[ll, k] == 1:
                                fst_find = True     
                                arrow_xmin = ll if arrow_xmin > ll else arrow_xmin
                                arrow_xmax = ll if arrow_xmax < ll else arrow_xmax
                            if arrow_map[ll, k] == 0 and fst_find:
                                fst_leave = True
                            if arrow_map[ll, k] == 1 and fst_leave:
                                sec_find = True

                        s_list.append(arrow_xmin)  # x 最小（南）
                        n_list.append(arrow_xmax)  # x 最大（北）
                        if sec_find:
                            dir_SN = False

                    if dir_SN:
                        if s_list[0] < max(s_list) and s_list[-1] < max(s_list):
                            arrow_direction = 'S'
                            pos = (max(s_list) - img.shape[0] // 2 + now_x,
                                    s_list.index(max(s_list)) + ymin - img.shape[1] // 2 + now_y)
                        if n_list[0] > min(n_list) and n_list[-1] > min(n_list):
                            arrow_direction = 'N'
                            pos = (min(n_list) - img.shape[0] // 2 + now_x,
                                   n_list.index(min(n_list)) + ymin - img.shape[1] // 2 + now_y)

                    # 考察新箭头中点和过去记录所有某个箭头中点的距离，如果有太近的就认为是图像抖动所致，反之判断为新箭头进行记录
                    if arrow_direction is not None:
                        redundant = False
                        for arrow in self.arrows:
                            if distance(arrow[0], pos) < 10:    
                                redundant = True
                        # 记录非冗余箭头
                        if not redundant:
                            self.arrows.append([pos, arrow_direction, True])

                    # 所有箭头两两比较，如果顶点距离 <60（认为是相邻箭头）且 ll 箭头在 k 箭头所指方向（意味着已经按 k 箭头指向走到了下一个箭头位置），则设置 ll 箭头为已经过箭头
                    for k in range(len(self.arrows)):
                        for ll in range(len(self.arrows)):
                            if k != ll and distance(self.arrows[k][0], self.arrows[ll][0]) < 60 and self.in_direction(
                                    self.arrows[ll], self.arrows[k][0]) > 0:
                                self.arrows[ll][2] = False  # 设为已经过箭头

        #print('self.arrows:',self.arrows)

    
    def get_angle_score(self, point) -> float:
        """ When agent heading to a target point, assign a score to the steering angle

        :param point: the target point that agent heading to
        :return: angle score
        :rtype: float

        :note: The score less than or equal to 0, the smaller the steering angle, the less score will be deducted
        """

        angle = direction(self.curr_pos(), point)   # 当前 agent 中心和 point 连线关于 x 负方向（南）的夹角
        angle_diff = angle - self.angle             # 相对当前 agent 绝对角度 self.angle 的转角
        angle_diff = angel_standardized(angle_diff) # 控制在 [-180,180]
        return -abs(angle_diff) / 180 * 0.5         


    def get_distance_score(self, point) -> float:
        """ When agent heading to a target point, assign a score to the distance

        :param point: the target point that agent heading to
        :return: distance score
        :rtype: float

        :note: The score less than or equal to 0, the closer the distance, the less score deducted
        """

        d = distance(self.curr_pos(), point)
        return -d / 50 if d > 70 else 0.

    def get_edges(self) -> Tuple[list,list]:
        """ Find all the passable edge in current global map, and select one of which to move forward based on some criteria
        
        :return: edge_list, target_bfs_path
        :rtype: edge_list is a list of dict, each element store the information of a edge, whose sturcture like {'points':[(x,y),(x,y)...],'center':(x,y)}, all the (x,y) are absolute coord
                target_bfs_path is a list of absolute coord, store the bfs path from traget edge center to current agent position

        :note: To pick up the target edge, factors such as arrow, distance and steering angle are taken into account 
                The center of a edge is the mean of the coords of all points of the edge
        """

        img = self.global_map
        edge_point_list = []                                                # 存储当前 global map 的所有探索边界点（都是绝对坐标）
        
        edge_map = np.zeros(img.shape[:2], dtype=int)                       # 记录地图中各个点的状态，是个 temp 变量
        past_x = np.ones(img.shape[:2], dtype=int) * -1                     # past_x[x,y] 记录 bfs 路径中 (x,y) 的前驱点 x 坐标 
        past_y = np.ones(img.shape[:2], dtype=int) * -1                     # past_y[x,y] 记录 bfs 路径中 (x,y) 的前驱点 y 坐标 
                                      
        agent_x = to_absolute_coord(self.x)                                 # agent 中心的中心坐标
        agent_y = to_absolute_coord(self.y)  

        # 从 agent 中心位置开始 bfs，找出当前已探索部分 global_map 的边界（可通行边界，墙壁不算）
        q = Queue()
        q.put((agent_x, agent_y))
        edge_map[agent_x, agent_y] = 1
        while not q.empty():
            x, y = q.get()
            
            # 如果 bfs 过程中找到了终点线像素，利用 past_x,past_y 反向找出从此位置到 agent 当前位置的逆序 bfs 路径返回
            if img[x, y, 2] > 0.1:  # 由于超分后图像会模糊，只要 img[x, y, 2] > 0.1，就认为 (x,y) 是终点线像素
                tx, ty = x, y
                target_bfs_path = [(x, y)]
                while tx != agent_x or ty != agent_y:
                    tx, ty = past_x[tx, ty], past_y[tx, ty]
                    target_bfs_path.append((tx, ty))
                return None, target_bfs_path

            # SENW 顺序 bfs 扩展，步进为 3 像素（应该是为了提高效率）
            for k in range(4):      
                nx = dx[k] * 3 + x 
                ny = dy[k] * 3 + y
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and edge_map[nx, ny] == 0:
                    
                    # 墙壁标记切片中已探索部分，img[x,y,1] 在 update 时置0了，edge_map 设 1
                    if img[nx, ny, 1] <= 0:
                        edge_map[nx, ny] = 1
                        q.put((nx, ny))
                    
                    # 墙壁标记切片中未探索部分，img[x,y,1] 还是 global_map 初始时的 150，边界点按 bfs 顺序存入 edge_point_list，edge_map 设 2
                    elif img[nx, ny, 1] >= 150:
                        edge_map[nx, ny] = 2
                        edge_point_list.append((nx, ny))
                    
                    # 用 past_x,past_y 记录 bfs 扩展路径
                    past_x[nx, ny] = x
                    past_y[nx, ny] = y

        #print(edge_point_list)

        # 把 edge_point_list 中的所有边界点划分为边（找出 agent 能走向的地方）
        edge_list = []
        for ep in edge_point_list:
            if edge_map[ep[0], ep[1]] == 2: # 确保 edge_point_list 中所有点都是墙壁标记切片的边界点
                center_x, center_y = 0, 0   # 一个边的中心点，它是该边所有点的 x,y 坐标均值
                edge = {'points': []}
                x, y = ep[0], ep[1]
                q = Queue()

                # 从点 ep 开始 bfs（步进为3），得到长度不超过 20 的 bfs 路径
                q.put((x, y))
                edge_map[x, y] = 3
                while not q.empty():
                    x, y = q.get()
                    edge['points'].append((x, y))
                    center_x += x
                    center_y += y
                    for k in range(8):
                        nx = dx[k] * 3 + x
                        ny = dy[k] * 3 + y
                        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and edge_map[nx, ny] == 2 and len(edge['points']) < 20:
                            edge_map[nx, ny] = 3
                            q.put((nx, ny))

                # bfs 路径边长度超过 4 就算一条边，center 为此边所有点的均值
                if len(edge['points']) > 4:
                    edge['center'] = (center_x / len(edge['points']), center_y / len(edge['points']))
                    edge_list.append(edge)

        #for edge in edge_list:
        #    print(edge)

        if len(edge_list) == 0:
            return None,None

        # 根据指向箭头数量、距离agent的距离和转向角度选出一个边
        reward = -100000
        target_edge = None
        for edge in edge_list:
            num = 0                                                 # 同向有效箭头数量
            dist = 1e10                                             # 同向最近有效箭头的距离
            for arrow in self.arrows:
                d = distance(edge['center'], arrow[0])              # 箭头中点和出边中点的距离
                flag = self.in_direction(arrow, edge['center'])     # 出边中点是否在箭头所指方向
                
                # 有效箭头，同向且距离小于 70，num 计数增加 1
                if arrow[2] and d < 70 and flag > 0:
                    num += 1
                    if dist > d + 1:
                        dist = d + 1

            num += self.get_angle_score(edge['center'])         # 去向该边转向角越小扣分越少
            num += self.get_distance_score(edge['center'])      # 距离该边越近扣分越少

            v = num if num < 0 else num / dist                  # 如果 num >= 0，使用 dist 对 num 打折，否则不动，计算出一条边的得分
            if reward < v:                                      # reward 记录最高得分，e 记录对应的边
                reward = v
                target_edge = edge

        # 选出边 target_edge 中距离最近该边 center 最近的点 point
        point = None
        dis = 1e10
        for p in target_edge['points']:
            d = distance(p, target_edge['center'])
            if dis > d:
                dis = d
                point = p

        # 返回从 point 到 agent 当前位置的 bfs 路径
        tx, ty = point[0], point[1]
        target_bfs_path = [(point[0], point[1])]
        while tx != agent_x or ty != agent_y:
            tx, ty = past_x[tx, ty], past_y[tx, ty]
            target_bfs_path.append((tx, ty))
        return edge_list,target_bfs_path

    def in_direction(self, arrow, point) -> int:
        """ Check whether the point is in the direction pointed by the arrow

        :param point: absolute coord of the point
        :param arrow: can be any arrow in self.arrows
        :return: return 1 means the point is in the direction pointed by the arrow
                 return -10 means the point is in the opposite direction pointed by the arrow
                 return 0 means the arrow and the point are not related (they are too far apart)
        :rtype: int
        """

        # point 和 arrow 顶点距离太近，认为同向（重合）返回 1 
        if distance(arrow[0], point) < 10:
            return 1

        dis_x = point[0] - arrow[0][0]
        dis_y = point[1] - arrow[0][1]

        if arrow[1] == 'E':
            if dis_y > 0 and abs(dis_x) < abs(dis_y) * 5:   # 同向返回 1
                return 1
            if dis_y < 0 and abs(dis_x) < abs(dis_y) * 5:   # 反向返回 -10
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

        return 0                                            # 没啥关系（point 和 arrow 相距太远）返回 0

    def fix_action(self, angle) -> float:
        """ Compensate for steering angle to steering early

        :param angle: steering angle
        :return: Compensated steering angle
        :rtype: float
        """

        if len(self.history) > 3:
            # 当前位置和 4 步之前位置连线的夹角，和当前 agent 绝对角度 self.angle 做差，把差距标准化到 [-180,180]
            a = direction(self.history[-4], self.curr_pos()) - self.angle
            a = angel_standardized(a)
            
            # 现在去往目标点的夹角 angle 在最大转向角度内（30），而过去一段时间的转向角度超过最大转向角度，且速度较快，就进行补偿
            if abs(angle) < 29 and abs(a) > 40 and self.v > 2.5:
                # 进行补偿
                new_angle = angle - a * min(abs(a), 20) / 8
        
                # 限制在有效取值范围内
                angle = angel_output_constrain(new_angle)

        return angle

    def get_angle_from_curr(self, point) -> float:
        """ Calculate the streeing angle to move forward to target point from agent's current position

        :param point: target point that agent heading to
        :return: aviliable streeing angle
        :rtype: float
        """

        dir = direction(self.curr_pos(), point) # 目标位置和南方夹角
        angle = dir - self.angle                # 减去 self.angle 得到转向角度

        angle = angel_standardized(angle)       # 标准化
        angle = angel_output_constrain(angle)   # 限制在有效取值范围内
    
        return angle

    def get_action(self, target_bfs_path) -> Tuple[float,float]:
        """ Calculate the streeing angle and output power to move forward to target point from agent's current position

        :param target_bfs_path: The bfs path from traget edge center to current agent position, which is a list form by absolute coords 
        :return: aviliable output power and streeing angle
        :rtype: Tuple[float,float]
        """

        # power * speed <= 400，保证能量不会耗尽
        power = 200 if self.v <= 2.01 else 200 / self.v * 2
        
        # 把 target_bfs_path 的最后一个点设为目标点，向 agent 当前位置连线，逐像素遍历，检查中间是否有墙壁
        # 如果有墙壁阻挡，就按 bfs 顺序回退，直到找到没有阻挡的点作为可行目标点，计算从当前位置去往可行目标点的转向角度
        agent_pos = self.curr_pos()
        for target in target_bfs_path[:-1]:
            dis = int(distance(target, agent_pos))
            aviliable = True
            for i in range(dis):
                x = int((target[0] - agent_pos[0]) / dis * i + agent_pos[0])
                y = int((target[1] - agent_pos[1]) / dis * i + agent_pos[1])
                if self.global_map[x, y, 1] > 0:    # 有墙壁
                    aviliable = False
            if aviliable:
                return power, self.get_angle_from_curr(target)

        return power, self.get_angle_from_curr(target)

    def act(self, obs) -> list:
        """ Calculate output actions based on raw observations

        :param obs: raw observations, shape = (25,25)
        :return: a list form as [force, angle], which will be driectly used to control agent

        :note: This is the key End-to-End function controlling the agent
        """
        # 包装原始观测
        obs_wrapped = self.wrap_obs(obs)                                
        
        if self.init:
            self.init = False
            force,angle = 0,0
            self.radius = 2.5

            # 初始时刻观测中点的中心坐标 (30,0)
            view_center_x = (self.radius + view_size) * scale_level     
            view_center_y = 0  

            # 把初始的包装观测拼接到 global map 中
            self.update_map(obs_wrapped, view_center_x, view_center_y)  

            # 更新 agent 圆周围 1.5 倍半径区域（清0）
            self.update_self_arround(1.5)                               

            #np.save('map_init.npy',self.global_map)

            # 识别当前观测中的箭头，记录到 self.arrows
            self.add_arrows(obs_wrapped[:, :, 0], 
                            to_absolute_coord(view_center_x), 
                            to_absolute_coord(view_center_y))   

            # 找出所有可去边，从中选出最好的边（综合考虑箭头、角度和距离），返回从 agent 当前位置到该边中点的 bfs 路径（逆序）
            self.draw_edges, target_bfs_path = self.get_edges()         
            self.draw_path = target_bfs_path
            #print(target_bfs_path)

            # 从目标点开始往 agent 当前位置遍历，直到找出和当前位置之间没有墙壁阻挡的点，计算去向该点的力和转向角度
            if target_bfs_path is None:
                force, angle = self.force, 0
            else:
                force, angle = self.get_action(target_bfs_path)         
        
        else:            
            # 基于上一帧 agent 中点坐标 (self.x,self.y) 计算观测中点的中心坐标
            view_center_x_last = self.x - math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
            view_center_y_last = self.y - math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level

            # 计算这一帧在 x 和 y 方向分别位移了多少
            displacement = get_displacement(obs_wrapped[:, :, 3],
                                    self.global_map[to_absolute_coord(int(view_center_x_last) - scaled_size // 2) : to_absolute_coord(int(view_center_x_last) + scaled_size // 2),
                                                    to_absolute_coord(int(view_center_y_last) - scaled_size // 2) : to_absolute_coord(int(view_center_y_last) + scaled_size // 2), 3],
                                    np.array([0, 0], dtype=np.float32))

            # 计算该帧观测中点的中心坐标
            view_center_x = view_center_x_last + displacement[0]
            view_center_y = view_center_y_last + displacement[1]

            # 计算该帧 agent 中点的中心坐标
            self.x = view_center_x + math.cos(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level
            self.y = view_center_y + math.sin(-self.angle / 180 * math.pi) * (self.radius + view_size) * scale_level

            # 更新 global_map
            self.update_map(obs_wrapped, view_center_x, view_center_y)
            
            # 更新 agent 周围区域
            self.update_self_arround()

            # 更新箭头
            self.add_arrows(obs_wrapped[:, :, 0], 
                            to_absolute_coord(view_center_x), 
                            to_absolute_coord(view_center_y)) 

            # 更新边列表和目标边
            self.draw_edges, target_bfs_path = self.get_edges()
            self.draw_path = target_bfs_path
            #print(target_bfs_path)

            # 计算去往目标边的力和转向角
            if target_bfs_path is None:
                force, angle = self.force, 0
            else:
                force, angle = self.get_action(target_bfs_path)
        
        # self.history 记录 agent 移动历史
        self.history.append(self.curr_pos())               

        # 位置差分计算 agent 绝对速度
        self.v = distance(self.history[-4], self.history[-1]) / 3 if len(self.history) > 3 else 1   
        
        # 转向补偿
        angle = self.fix_action(angle)                      
        self.angle += angle # 更新当前 agent 的绝对角度
        self.angle = angel_standardized(self.angle)
        self.force = force
        
        #np.save('map11.npy',self.global_map)
        #self.draw_global_map()

        return [force, angle]

    def draw_global_map(self,x=-1,y=-1) -> None:
        """ Draw current global map (just for debug)
        
        :param x,y: the absolute coord of a debug point, it will be drawn prominently to help you to debug
        :return: None

        :note: you can call this function at any position of the code, then current global map, aviliable edges and the bfs path of target edge will be drawn clearly,
               those information can help you to understand how the program is running
        """
        map = self.global_map.copy()
        map[map[:,:,3] == 60] = np.array([0,0,0,100])
        map[map[:,:,3] == 120] = np.array([0,0,0,255])
        map[map[:,:,3] == 150] = np.array([0,0,0,50])
        
        if self.draw_edges != None:
            for edge in self.draw_edges:
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                for i,point in enumerate(edge['points']):
                    map[point[0],point[1]] = np.array([color[0],color[1],color[2],i*255/len(edge['points'])])
                map[int(edge['center'][0]),int(edge['center'][1])] = np.array([0,0,0,255])

        if self.draw_path != None:
            for i,point in enumerate(self.draw_path):
                map[point[0],point[1]] = np.array([100,100,50,i*255/len(self.draw_path)])

        if x != -1 and y != -1:
            map[x,y] = np.array([255,0,0,100])
            map[x-1,y] = np.array([255,0,0,100])
            map[x+1,y] = np.array([255,0,0,100])
            map[x,y-1] = np.array([255,0,0,100])
            map[x,y+1] = np.array([255,0,0,100])

        pic = Image.fromarray(np.uint8(map))
        pic.show()

    def choose_action(self,observation) -> None:
        """ For local evaluation script

        note: This function will only be called in root/evaluation_local.py 
        """
        actions = self.act(observation['obs'])
        wrapped_actions = [[actions[0]], [actions[1]]]
        return wrapped_actions

agent = RuleAgent()
agent.reset()

def my_controller(observation, action_space, is_act_continuous=False) -> None:
    """ For online evaluation script

        note: This function will be called in jidi platform to evaluate agent performence. 
              It will also be called in root/run_log.py, which is very similar to the online evaluation script, so that you can check your agent's performence easily 
    """
    #actions = agent.act(observation)
    actions = agent.act(observation['obs'])
    wrapped_actions = [[actions[0]], [actions[1]]]
    return wrapped_actions
