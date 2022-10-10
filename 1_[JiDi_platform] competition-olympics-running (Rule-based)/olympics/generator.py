import json
import sys
import os
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(father_path))

module = __import__("object")

def create_scenario(scenario_name):
    file_path = os.path.join(os.path.dirname(__file__), 'maps.json')
    with open(file_path) as f:
        conf = json.load(f)[scenario_name]

    GameMap = dict()
    GameMap["objects"] = list()
    GameMap["agents"] = list()
    GameMap["view"] = conf["view"]

    for type in conf:
        if (type == "wall") or (type == "cross"):
            #print("!!", conf[type]["objects"])
            for key, value in conf[type]["objects"].items():
                GameMap["objects"].append(getattr(module, type.capitalize())
                     (
                     init_pos=value["initial_position"],
                    length=None,
                     color=value["color"],
                     ball_can_pass = value['ball_pass'] if ("ball_pass" in value.keys()
                                                            and value['ball_pass']=="True") else False
                 )
                 )
        elif type == 'arc':
            for key, value in conf[type]['objects'].items():
                #print("passable = ", bool(value['passable']))
                GameMap['objects'].append(getattr(module, type.capitalize())(
                    init_pos = value["initial_position"],
                    start_radian = value["start_radian"],
                    end_radian = value["end_radian"],
                    passable = True if value["passable"] == "True" else False,
                    color = value['color'],
                    collision_mode = value['collision_mode']
                ))

        elif type in ["agent","ball"]:
            for key, value in conf[type]["objects"].items():
                GameMap["agents"].append(getattr(module, type.capitalize())
                     (
                     mass=value["mass"],
                     r=value["radius"],
                     position=value["initial_position"],
                    color=value["color"]
                 ),
                                           )
    # print(" ========================== check GameMap ==========================")
    #print(GameMap)
    return GameMap