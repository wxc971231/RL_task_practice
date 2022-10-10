- 赛题：[RLChina2021智能体竞赛](http://www.jidiai.cn/compete_detail?compete=12)
- 复现详细介绍：[RL 实践（0）—— 及第平台辛丑年冬赛季【Rule-based policy】](https://blog.csdn.net/wxc971231/article/details/125438242)
- 该项目引用并改编自 [官方 Repo](https://github.com/jidiai/Competition_Olympics-Running)，以下为其原始 README 文件

# Competition_1v1running

## Environment


### “奥林匹克 跑步运动”:

<b>标签：</b>不完全观测；连续动作空间；连续状态空间

<b>环境简介：</b>智能体参加奥林匹克运动会。在这个系列的竞赛中，
两个智能体参加跑步竞赛，目标是尽快到达终点。

<b>环境规则:</b> 
1. 在一个随机地图中，对战双方各控制一个有相同质量和半径的弹性小球智能体；
2. 智能体可以互相碰撞，也可以碰撞墙壁，但会损失一定的速度；
3. 智能体自身有能量，每步消耗的能量与施加的驱动力和位移成正比；
4. 智能体能量以固定速率恢复，如果能量衰减到零，则不能加力；
5. 智能体的观测为自身朝向前方25*25的矩形区域，观测值包括墙壁、终点线、对手和跑道方向辅助箭头；
6. 初始时智能体位于所在地图的起跑线位置，初始朝向与跑道方向平行；
7. 当有一个智能体到达终点（红线）或环境达到最大步数500步时环境结束，先冲过终点的一方获胜，若双方均未过线则平局；
8. 智能体需要具有一定的泛化性以适应不同的地图，评测时会从所有地图（每次热身赛和正赛评测时可能加入新地图）中随机选择一个作为评测地图。

<b>动作空间：</b>连续；两维。分别代表施加力量和转向角度。

<b>观测：</b>每一步环境返回一个25x25的二维矩阵，详情请见*/olympics*文件夹

<b>奖励函数:</b> 如果没到达终点，不得分；如果到达终点，获得100分。

<b>环境终止条件:</b> 有一个智能体到达终点，则环境结束；或者，环境达到最大步数500步。

<b>评测说明：</b>该环境属于零和游戏，在金榜的积分按照ELO进行匹配算法进行计算并排名。零和游戏在匹配对手或队友时，按照瑞士轮进行匹配。
平台验证和评测时，在单核CPU上运行用户代码（暂不支持GPU），限制用户每一步返回动作的时间不超过1s，内存不超过500M。

<b>报名方式：</b>访问“及第”平台（ www.jidiai.cn ），在“擂台”页面选择“RLChina 智能体挑战赛 - 辛丑年冬赛季”即可报名参赛。RLCN 微信公众号后台回复“智能体竞赛”，可进入竞赛讨论群。

This is a POMDP simulated environment of 2D sports games where althletes are spheres and have continuous action space (torque and steering). The observation is a 25*25 array of agent's limited view range. We introduce collision and agent's fatigue such that no torque applies when running out of energy.

This is for now a beta version and we intend to add more sports scenario, stay tuned :)

---
## Dependency

>conda create -n olympics python=3.8.5

>conda activate olympics

>pip install -r requirements.txt

---

## Run a game

>python olympics/main.py

---

## Train a baseline agent 

>python rl_trainer/main.py

By default parameters, the total reward of training is shown below.

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/PPO%20map1%20training%20(run1).png>

You can also locally evaluate your trained model by executing:

>python evaluation_local.py --my_ai rl --opponent random --episode=50 --map=all

or specifying the map number (--map=1).

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/evaluation_local_results.png>


## How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example,

>python run_log.py --my_ai "rl" --opponent "random"

in which you are controlling agent 1 which is green.

## Ready to submit

1. Random policy --> *agents/random/submission.py*
2. RL policy --> *all files in agents/rl*
