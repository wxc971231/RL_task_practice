import numpy as np
import time
import matplotlib.pyplot as plt
from Brain import Brain
from Setting import MAX_CARS

class Agent:
    def __init__(self,env):
        self.env = env
        self.brain = Brain(self)
        self.timeStep = 0.1

    def getBrain(self):
        return self.brain

    # 系统复位
    def reset(self):
        self.brain.reset()
        self.env.updateUILabel('')
        self.env.updateUI()

    # 生成图表
    def genFigures(self):
        self.addFigure(self.brain.getPolicy(), ['num of cars in first location', 'num of cars in second location', '# of cars to move during night'],1)
        self.addFigure(self.brain.getValueTable(), ['num of cars in first location', 'num of cars in second location', 'expected returns'],2)
        plt.show()

    # 设置新图表
    def addFigure(self,data,labels,index):
        fig = plt.figure(index)
        ax = fig.add_subplot(111, projection='3d')
        AxisZ = []
        AxisXPrint = []
        AxisYPrint = []
        for i, j in self.env.getStateIndex():
            AxisXPrint.append(i)
            AxisYPrint.append(j)
            AxisZ.append(data[i, j])
        ax.scatter(AxisXPrint, AxisYPrint, AxisZ)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

    # 设置计算延时周期
    def setTimeStep(self,timeStep):
        self.timeStep = timeStep

    # 训练方法 ---------------------------------------------------------------------------
    # 执行一次策略评估
    def policyEvaluationOneStep(self):
        sumValue_ = self.brain.getSumValue()
        newStateValue = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        for i,j in self.env.getStateIndex():
            a = self.brain.getPolicy()[i,j]
            newStateValue[i,j] = self.env.getValue([i,j],a,self.brain.getValueTable())
        self.brain.setValueTabel(newStateValue)

        valueChanged = abs(self.brain.getSumValue()-sumValue_)
        self.env.updateUILabel('value changed: {}'.format(round(valueChanged,2)))
        self.env.updateUI()
        return valueChanged
        
    # 执行一次策略提升
    def policyImporve(self):
        newPolicy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        actions = self.brain.getActions()
        for i, j in self.env.getStateIndex():
            actionReturns = []
            # go through all actions and select the best one
            for a in actions:
                if (a >= 0 and i >= a) or (a < 0 and j >= abs(a)):
                    q = self.env.getValue([i,j],a,self.brain.getValueTable())
                    actionReturns.append(q)
                else:
                    actionReturns.append(-float('inf'))
            bestAction = np.argmax(actionReturns)    
            newPolicy[i, j] = actions[bestAction]      

        policyChanged = np.sum(newPolicy != self.brain.getPolicy()) != 0
        self.brain.setPolicy(newPolicy)
            
        self.env.updateUILabel('')
        self.env.updateUI()
        return policyChanged

    # 反复评估当前策略
    def policyEvaluation(self):
        while self.brain.getAutoExec() == True:
            self.policyEvaluationOneStep()
            time.sleep(self.timeStep)

    # 策略迭代算法
    def policyIteration(self):
        policyChanged = True
        while self.brain.getAutoExec() == True and policyChanged:
            # 反复评估直到当前策略下价值收敛
            valueChanged = self.policyEvaluationOneStep()
            while self.brain.getAutoExec() == True and valueChanged > 0.5:
                valueChanged = self.policyEvaluationOneStep()
                time.sleep(self.timeStep) 
            # 执行一次策略提升
            policyChanged = self.policyImporve()

    # 价值迭代算法（等价于 policyImporve 和 policyEvaluationOneStep 交替运行）        
    def valueIteration(self):
        policyChanged = True
        valueChanged = 1
        # 这里我希望手动停止,所以条件中没有 policyChanged
        while self.brain.getAutoExec() == True and valueChanged > 0.5:
            valueChanged = self.policyEvaluationOneStep()
            policyChanged = self.policyImporve()    
            time.sleep(self.timeStep)

