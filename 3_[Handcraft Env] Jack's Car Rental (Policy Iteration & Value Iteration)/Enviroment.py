import math
import numpy as np
from Agent import Agent
from UI import EnviromentUI

from Setting import *

class Enviroment:
    def __init__(self):
        self.mainUI = None
        self.agent = Agent(self)
        self.envUI = EnviromentUI(self)

        temp1 = np.arange(MAX_CARS+1)
        temp2 = np.arange(MAX_CARS+1)
        temp2,temp1 = np.meshgrid(temp2,temp1)
        self.stateIndex = np.vstack([temp1.ravel(), temp2.ravel()]).T

    def setMainUI(self,mainUI):
        self.mainUI = mainUI

    def getAgent(self):
        return self.agent
    
    def updateUILabel(self,label):
        if self.mainUI != None:
            self.mainUI.label_info.setText(label)

    def updateUI(self):
        self.envUI.update()

    def getEnvUI(self):
        return self.envUI

    def getStateIndex(self):
        return self.stateIndex

    # 当前价值表为 stateValue，计算给定 state 执行 action 的 TD target
    def getValue(self, state, action, stateValue):
        # init return
        returns = 0.0
        returns -= MOVE_CAR_COST * abs(action)

        # 可行出借数组
        numOfCarsFirstLoc = int(min(state[0] - action, MAX_CARS))
        numOfCarsSecondLoc = int(min(state[1] + action, MAX_CARS))

        # temp
        num1 = np.arange(POISSON_UP_BOUND)
        num2 = np.arange(POISSON_UP_BOUND)
        num2,num1 = np.meshgrid(num2,num1)

        # 出借数量矩阵    
        numReq = np.vstack([num2.ravel(), num1.ravel()]).T
        numReq.resize((POISSON_UP_BOUND,POISSON_UP_BOUND,2))
        numReq[numReq[:,:,0]>numOfCarsFirstLoc,0] = numOfCarsFirstLoc
        numReq[numReq[:,:,1]>numOfCarsSecondLoc,1] = numOfCarsSecondLoc
        
        # 获取租金
        reward = np.sum(numReq, axis=2)*RENTAL_CREDIT

        # 出租后剩余车数
        num = [numOfCarsFirstLoc,numOfCarsSecondLoc] - numReq

        # constantReturnedCars = True 则把换车数量从泊松分布简化为定值
        constantReturnedCars = True
        if constantReturnedCars:         
            # 所有可达新状态
            num += [RETURNS_FIRST_LOC,RETURNS_SECOND_LOC]
            num[num>MAX_CARS] = MAX_CARS

            # 新状态价值
            values = stateValue[num[:,:,0],num[:,:,1]]

            # 计算收益
            returns += np.sum(RENTAL_PROB * (reward + DISCOUNT * values))
        else:
            # matrix of number of cars returned   
            numRet = np.vstack([num1.ravel(), num2.ravel()]).T
            numRet.resize((POISSON_UP_BOUND,POISSON_UP_BOUND,2))

            for returnedCarsFirstLoc in range(0, POISSON_UP_BOUND):
                for returnedCarsSecondLoc in range(0, POISSON_UP_BOUND):
                    # viable new states
                    num += [returnedCarsFirstLoc,returnedCarsSecondLoc]
                    num[num>MAX_CARS] = MAX_CARS

                    # values of viable states
                    values = stateValue[num[:,:,0],num[:,:,1]]

                    returns += np.sum(RENTAL_PROB * RETURN_PROB[returnedCarsFirstLoc,returnedCarsSecondLoc]*(reward + DISCOUNT * values))
        return returns

            


