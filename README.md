# About this repository

- I decided to upload all my RL task demo here, include
  1. Project code written while learning
  2. Code written for some competition
  3. Anything about RL task... I don‘t know
- Anyway, I hope every project here is complete, and can be run or train directly to solve an individual RL task
- The project name will follow this format: `[env_name] task_name (method_name/method_type)`



# Demo List

## 1. [JiDi_platform] competition-olympics-running (Rule-based)

- **Project type**: A recurrence project for [an RL competition on JiDi AI platform](http://www.jidiai.cn/compete_detail?compete=12)
- **Raw Champion code**: [Luanshaotong/Competition_Olympics-Running](https://gitee.com/luanshaotong/competition-olympics-running/blob/lst/olympics/submission.py)

- **Detailed description**: [RL 实践（0）—— 及第平台辛丑年冬赛季【Rule-based policy】](https://blog.csdn.net/wxc971231/article/details/125438242)



## 2. [Handcraft Env] K-arms bandit (MC)

- **Project type**:  Compare the performance of four simple ways to balance exploration and exploitation in K-arms bandit environment, include

  1. $\epsilon$-greedy
  2. Decaying $\epsilon$-greedy
  3. Upper confidence bound (UCB)
  4. Thompson sampling

  note that K-arms bandit environment is a simplified version of RL paradigm without state transform

- **Detailed description**: [RL 实践（1）—— 多臂赌博机](https://blog.csdn.net/wxc971231/article/details/127103190)



## 3. [Handcraft Env] Jack's Car Rental (Policy Iteration & Value Iteration)

- **Project type**:  Implement an example of 《Reinforcement Learning An Introduction》with GUI

  <img src="img/Jack's_car_rental.png" style="zoom:75%;" />

- **Detailed description**: [RL 实践（2）—— 杰克租车问题【策略迭代 & 价值迭代】](https://blog.csdn.net/wxc971231/article/details/127222242)

...be continued