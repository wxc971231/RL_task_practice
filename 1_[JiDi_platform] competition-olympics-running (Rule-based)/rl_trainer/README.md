## training

Train a PPO agent:

>python main.py --algo=ppo --map=1

or train on different map:

>python main.py --algo=ppo --shuffle_map

For your reference, we have trained a vanilla PPO agent independently on four given maps and the training reward plot is shown as:

### Map 1

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/PPO%20map1%20training%20(run1).png>

### Map 2

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/PPO%20map2%20training%20(run2).png>

### Map 3

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/PPO%20map3%20training%20(run3).png>

### Map 4

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/PPO%20map4%20training%20(run4).png>

(we assume the opponent is not moving at all.)




# Evaluating

>python main.py --load_model --algo=ppo --map=1 --load_run=1 --load_episode=900
>
>python main.py --load_model --algo=ppo --map=2 --load_run=2 --load_episode=900
>
>python main.py --load_model --algo=ppo --map=3 --load_run=3 --load_episode=900
>
>python main.py --load_model --algo=ppo --map=4 --load_run=4 --load_episode=1500



