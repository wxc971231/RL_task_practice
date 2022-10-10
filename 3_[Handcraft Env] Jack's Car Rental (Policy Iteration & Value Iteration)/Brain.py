import numpy as np
import threading
from Setting import MAX_CARS,MAX_MOVE_OF_CARS

class Brain:
    def __init__(self,agent):
        self.agent = agent

        self.actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)
        self.valueTable = np.zeros((MAX_CARS+1,MAX_CARS+1))
        self.QTable = np.zeros((MAX_CARS+1,MAX_CARS+1,self.actions.size))
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1),dtype='int')

        self.autoExec = False   # 自动执行
        self.autoThread = None  # 自动执行线程

    def getActions(self):
        return self.actions

    def getValueTable(self):
        return self.valueTable

    def setValueTabel(self,table):
        self.valueTable = table

    def getMaxValue(self):
        return float(np.max(self.valueTable))

    def getMinValue(self):
        return float(np.min(self.valueTable))

    def getMaxAction(self):
        return float(np.max(self.policy))

    def getMinAction(self):
        return float(np.min(self.policy))
    
    def getSumValue(self):
        return float(np.sum(self.valueTable))

    def getQTale(self):
        return self.QTable
    
    def setQTable(self,row,colum,action,q):
        self.QTable[row,colum,action] = q

    def getPolicy(self):
        return self.policy
    
    def setPolicy(self,policy):
        self.policy = policy

    def getAutoExec(self):
        return self.autoExec
    
    def setAutoExec(self,exec):
        self.autoExec = exec

    def reset(self):
        self.autoExec = False
        self.waitAutoExecEnd()
        self.valueTable = np.zeros((MAX_CARS+1,MAX_CARS+1))
        self.QTable = np.zeros((MAX_CARS+1,MAX_CARS+1,self.actions.size))
        self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1),dtype='int')
        self.agent.env.mainUI.pbt_autoEvaluation.setEnabled(True)
        self.agent.env.mainUI.pbt_policyIteration.setEnabled(True)
        self.agent.env.mainUI.pbt_valueIteration.setEnabled(True)

    # 结束策略自动执行子线程
    def waitAutoExecEnd(self):
        self.autoExec = False
        if self.autoThread != None:
            while self.autoThread.is_alive():
                pass

    def autoToggle(self,mode):
        if self.autoExec:
            self.autoExec = False
            self.agent.env.mainUI.pbt_autoEvaluation.setEnabled(True)
            self.agent.env.mainUI.pbt_policyIteration.setEnabled(True)
            self.agent.env.mainUI.pbt_valueIteration.setEnabled(True)
        else:   # 启动子线程自动执行策略
            self.autoExec = True
            if mode == 'policy iteration':
                self.agent.env.mainUI.pbt_autoEvaluation.setEnabled(False)
                self.agent.env.mainUI.pbt_valueIteration.setEnabled(False)
                self.autoThread = threading.Thread(target = self.agent.policyIteration)
            elif mode == 'policy evaluation':
                self.agent.env.mainUI.pbt_policyIteration.setEnabled(False)
                self.agent.env.mainUI.pbt_valueIteration.setEnabled(False)
                self.autoThread = threading.Thread(target = self.agent.policyEvaluation)
            elif mode == 'value iteration':
                self.agent.env.mainUI.pbt_policyIteration.setEnabled(False)
                self.agent.env.mainUI.pbt_autoEvaluation.setEnabled(False)
                self.autoThread = threading.Thread(target = self.agent.valueIteration)
            else:
                assert False
            self.autoThread.setDaemon(True)
            self.autoThread.start()    