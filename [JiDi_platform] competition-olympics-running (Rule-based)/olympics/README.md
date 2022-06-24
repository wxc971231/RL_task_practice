## Olympics engine

Here lies the underlying physics engine for the running environment, including the collision detection and response, map generation and pygame visualisation.

To play a game, run the *main.py* file or execute:

> python main.py


### Partial observation

In every map, agents can only see object in front of him, this includes the bouncable wall, crossable arrow, opponent agents and the Final.

The observation is a 25*25 array shown as below:

<img src=https://gitee.com/jidiai/competition-olympics-running/raw/main/assets/agent_view.png>

where each entry corresponds to different object, for instance:

| Entry value  |    object    |
|--------------|-------------:|
|     1        | agent green  |
|     4        |    arrow     |
|     5        | agent purple |
|     6        |    wall      |
|     7        |    Final     |




