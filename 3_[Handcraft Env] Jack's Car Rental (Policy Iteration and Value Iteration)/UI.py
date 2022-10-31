import numpy as np
import math
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter,QColor,QFont,QPen,QBrush
from numpy.lib.function_base import bartlett
from Agent import Agent
from Setting import MAX_CARS

def valueLimit(value,MAX,MIN):
    if value > MAX: return MAX
    if value < MIN: return MIN
    return value

# 窗口 UI
class MainWindowUI(QtWidgets.QMainWindow):
    def __init__(self,env):
        super().__init__()
        self.env = env
        self.env.setMainUI(self)
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(622, 612)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.checkBox_viewAction = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_viewAction.setText("")
        self.checkBox_viewAction.setObjectName("checkBox_viewAction")
        self.gridLayout.addWidget(self.checkBox_viewAction, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.checkBox_viewValue = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_viewValue.setText("")
        self.checkBox_viewValue.setObjectName("checkBox_viewValue")
        self.checkBox_viewValue.setChecked(True)
        self.checkBox_viewValue.setEnabled(False)
        self.gridLayout.addWidget(self.checkBox_viewValue, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.spinBox_timeStep = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.spinBox_timeStep.setObjectName("spinBox_timeStep")
        self.spinBox_timeStep.setSingleStep(0.01)
        self.spinBox_timeStep.setValue(0.05)
        self.gridLayout.addWidget(self.spinBox_timeStep, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pbt_evaluation = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_evaluation.setObjectName("pbt_evaluation")
        self.gridLayout_2.addWidget(self.pbt_evaluation, 0, 0, 1, 1)
        self.pbt_improvement = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pbt_improvement.sizePolicy().hasHeightForWidth())
        self.pbt_improvement.setSizePolicy(sizePolicy)
        self.pbt_improvement.setObjectName("pbt_improvement")
        self.gridLayout_2.addWidget(self.pbt_improvement, 0, 1, 2, 1)
        self.pbt_autoEvaluation = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_autoEvaluation.setObjectName("pbt_autoEvaluation")
        self.gridLayout_2.addWidget(self.pbt_autoEvaluation, 1, 0, 1, 1)
        self.pbt_policyIteration = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_policyIteration.setObjectName("pbt_policyIteration")
        self.gridLayout_2.addWidget(self.pbt_policyIteration, 2, 0, 1, 2)
        self.pbt_valueIteration = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_valueIteration.setObjectName("pbt_valueIteration")
        self.gridLayout_2.addWidget(self.pbt_valueIteration, 3, 0, 1, 2)
        self.pbt_reset = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_reset.setObjectName("pbt_reset")
        self.gridLayout_2.addWidget(self.pbt_reset, 4, 0, 1, 2)
        self.pbt_figure = QtWidgets.QPushButton(self.centralwidget)
        self.pbt_figure.setObjectName("pbt_figure")
        self.gridLayout_2.addWidget(self.pbt_figure, 5, 0, 1, 2)
        self.label_info = QtWidgets.QLabel(self.centralwidget)
        self.gridLayout_2.addWidget(self.label_info, 6, 0, 1, 2)
        self.verticalLayout.addLayout(self.gridLayout_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_3.addLayout(self.verticalLayout, 2, 4, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_3.addWidget(self.line_3, 1, 0, 2, 1)
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout_3.addWidget(self.line_5, 3, 1, 1, 4)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_3.addWidget(self.line_2, 2, 3, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_3.addWidget(self.line_4, 0, 0, 2, 5)
        self.widget = self.env.getEnvUI()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(600, 600))
        self.widget.setObjectName("widget")
        self.gridLayout_3.addWidget(self.widget, 2, 1, 1, 2)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 622, 23))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.connectSignalAndSlot()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "查看动作"))
        self.label_3.setText(_translate("MainWindow", "查看价值"))
        self.label.setText(_translate("MainWindow", "计算周期"))
        self.label_2.setText(_translate("MainWindow", "折扣系数"))
        self.pbt_evaluation.setText(_translate("MainWindow", "策略评估"))
        self.pbt_improvement.setText(_translate("MainWindow", "策略提升"))
        self.pbt_autoEvaluation.setText(_translate("MainWindow", "自动评估"))
        self.pbt_policyIteration.setText(_translate("MainWindow", "策略迭代"))
        self.pbt_valueIteration.setText(_translate("MainWindow", "价值迭代"))
        self.pbt_reset.setText(_translate("MainWindow", "复位"))
        self.pbt_figure.setText(_translate("MainWindow", "出图"))

    def connectSignalAndSlot(self):
        self.pbt_evaluation.clicked.connect(self.env.agent.policyEvaluationOneStep)
        self.pbt_autoEvaluation.clicked.connect(lambda:self.env.agent.brain.autoToggle('policy evaluation'))
        self.pbt_policyIteration.clicked.connect(lambda:self.env.agent.brain.autoToggle('policy iteration'))
        self.pbt_valueIteration.clicked.connect(lambda:self.env.agent.brain.autoToggle('value iteration'))
        self.pbt_reset.clicked.connect(self.env.agent.reset)
        self.pbt_improvement.clicked.connect(self.env.agent.policyImporve)
        self.pbt_figure.clicked.connect(self.env.agent.genFigures)
        self.spinBox_timeStep.valueChanged.connect(lambda:self.env.agent.setTimeStep(float(self.spinBox_timeStep.value())))
        self.checkBox_viewValue.clicked.connect(lambda:self.setShowMode('value'))
        self.checkBox_viewAction.clicked.connect(lambda:self.setShowMode('action'))
        

    def setShowMode(self,mode):
        self.env.getEnvUI().setShowMode(mode)
        self.checkBox_viewValue.setEnabled(mode != 'value')
        self.checkBox_viewValue.setChecked(mode == 'value')
        self.checkBox_viewAction.setEnabled(mode == 'value')
        self.checkBox_viewAction.setChecked(mode != 'value')
        self.env.updateUI()

# 状态方格 UI 类
class StateUI:
    def __init__(self,x,y,row,colum,l,envUI):
        self.agent = envUI.env.getAgent()
        self.envUI = envUI
        self.x = x              # 左上角坐标
        self.y = y
        self.l = l              # 边长
        self.centerX = x+0.5*l  # 中心坐标
        self.centerY = y+0.5*l
        self.row = row          # 行位置
        self.colum = colum      # 列位置

    def resize(self,x,y,l):
        self.x = x
        self.y = y
        self.l = l
        self.centerX = x+0.5*l  # 中心坐标
        self.centerY = y+0.5*l

    def draw(self,painter):
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))  
        value = self.agent.getBrain().getValueTable()[self.row,self.colum]
        action = self.agent.getBrain().getPolicy()[self.row,self.colum]

        # 显示设置
        if self.envUI.getShowMode() == 'value':
            temp = float(value)
            maxTemp = self.agent.getBrain().getMaxValue()
            minTemp = self.agent.getBrain().getMinValue()

            T = 0 if maxTemp == minTemp else valueLimit(255*(temp-minTemp)/(maxTemp-minTemp),255,0)
            if temp <= 0:
                brush = QBrush(QColor(T,0,0),Qt.SolidPattern) 
            else:
                brush = QBrush(QColor(0,T,0),Qt.SolidPattern) 
        else:
            temp = float(action)
            maxTemp = self.agent.getBrain().getMaxAction()
            minTemp = self.agent.getBrain().getMinAction()

            if temp == 0:
                brush = QBrush(QColor(230,230,230),Qt.SolidPattern) 
            elif temp < 0:
                R = valueLimit(255*(temp-minTemp)/(0-minTemp),255,0)
                brush = QBrush(QColor(R,0,0),Qt.SolidPattern) 
            else:
                G = valueLimit(255*(temp-0)/(maxTemp-0),255,0)
                brush = QBrush(QColor(0,G,0),Qt.SolidPattern) 
        
        # 绘制
        painter.setBrush(brush)
        painter.drawRect(self.x, self.y, self.l, self.l)

        # 叠加文本信息
        painter.setPen(QColor(100,100,100))
        painter.setFont(QFont('微软雅黑', 0.2*self.l))
        painter.drawText(QRect(self.x+0.1*self.l, self.y+0.1*self.l, 0.8*self.l, 0.8*self.l),Qt.AlignRight | Qt.AlignTop,str(round(value,2)))
        painter.drawText(QRect(self.x+0.1*self.l, self.y+0.1*self.l, 0.8*self.l, 0.8*self.l),Qt.AlignLeft | Qt.AlignBottom,str(action))

        #painter.setFont(QFont('微软雅黑', 0.15*self.l))
        #painter.drawText(QRect(self.x+0.1*self.l, self.y+0.1*self.l, 0.8*self.l, 0.8*self.l),Qt.AlignHCenter | Qt.AlignBottom,str(self.colum))
        #painter.drawText(QRect(self.x+0.1*self.l, self.y+0.1*self.l, 0.8*self.l, 0.8*self.l),Qt.AlignVCenter | Qt.AlignLeft,str(self.row))

# 环境 UI 类（所有状态方格组成）
class EnviromentUI(QWidget):
    def __init__(self,env):
        super().__init__() 
        self.env = env

        self.rowNum = MAX_CARS+1          
        self.columNum = MAX_CARS+1
        self.painter = QPainter(self)      

        self.mousePos = QtCore.QPoint()     # 当前光标位置
        self.cornerPos = QtCore.QPoint()    # 左上角坐标

        self.zoom = 1       # 放缩
        self.cubeL = 0      # 方格边长
        self.inited = False 

        self.setMouseTracking(True)         # 允许鼠标追踪事件
        
        self.showMode = 'value' # 'value' or 'action'

        self.states = []
        for r in range(self.rowNum):
            columStates = []
            for c in range(self.columNum):
                columStates.append(StateUI(0,0,r,c,0,self))
            self.states.append(columStates)
        
    def getShowMode(self):
        return self.showMode
    
    def setShowMode(self,mode):
        self.showMode = mode
    
    # 绘制事件
    def paintEvent(self,event):
        self.painter.begin(self)
        self.paintEvent = event
        self.drawGrid(self.painter)
        self.painter.end()

    # 鼠标追踪事件
    def mouseMoveEvent(self,event):
        self.mousePos = event.pos()

    # 鼠标滚轮滚动事件
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()              # 竖直滚过的距离
        if angleY > 0:
            self.zoom += 0.1
        else:  
            self.zoom -= 0.1
            if self.zoom < 1:
                self.zoom = 1

        self.update()

    # 绘制地图网格  
    def drawGrid(self,painter):
        width = self.width() + 2*self.cubeL 
        height = self.height() - 2*self.cubeL
        
        # 目前光标所在方格坐标,缩放时用作不动点
        if self.cubeL != 0:
            last_colum = math.floor((self.mousePos.x()-self.cornerPos.x())/self.cubeL)
            last_row = math.floor((self.mousePos.y()-self.cornerPos.y())/self.cubeL)
            last_colum = valueLimit(last_colum,self.columNum-1,0)
            last_row = valueLimit(last_row,self.rowNum-1,0)
            
        # 以较长边为基础确定方格边长
        if self.columNum/self.rowNum >= width/height:
            self.cubeL = self.zoom*width/self.columNum
        else:
            self.cubeL = self.zoom*height/self.rowNum
        
        # 缩放比大于1，以鼠标所在方格左上角为不动点缩放；缩放比等于1且继续下滚时，逐渐调整方格区至中部完全显示位置
        if self.zoom > 1:
            self.cornerPos.setX(self.states[last_row][last_colum].x - self.cubeL*last_colum)
            self.cornerPos.setY(self.states[last_row][last_colum].y - self.cubeL*last_row)  
        else:
            cornerPos = QtCore.QPoint()
            if self.columNum/self.rowNum >= width/height:
                cornerPos.setX(0)
                cornerPos.setY(0.5*height - 0.5*self.rowNum*self.cubeL)
            else:
                cornerPos.setX(0.5*width - 0.5*self.columNum*self.cubeL)
                cornerPos.setY(0)

            if self.inited:
                self.cornerPos.setX(0.7*self.cornerPos.x() + 0.3*cornerPos.x())
                self.cornerPos.setY(0.7*self.cornerPos.y() + 0.3*cornerPos.y())
            else:
                self.inited = True   
                self.cornerPos = cornerPos

        # 刷新cubes二维列表并绘制
        for r in range(self.rowNum):
            for c in range(self.columNum):
                self.states[r][c].resize(self.cornerPos.x()+c*self.cubeL, self.cornerPos.y()+r*self.cubeL, self.cubeL)
                self.states[r][c].draw(painter)

        # 分割线 & 刻度
        painter.setFont(QFont('微软雅黑', 0.3*self.cubeL))
        for r in range(self.rowNum):
            painter.drawLine(self.cornerPos.x(), self.cornerPos.y()+self.cubeL*r, self.cornerPos.x()+self.cubeL*self.columNum, self.cornerPos.y()+self.cubeL*r)
            painter.drawText(QRect(self.cornerPos.x()-self.cubeL, self.cornerPos.y()+self.cubeL*r, self.cubeL, self.cubeL),Qt.AlignCenter,str(r))

        for c in range(self.columNum):
            painter.drawLine(self.cornerPos.x()+c*self.cubeL, self.cornerPos.y(), self.cornerPos.x()+c*self.cubeL, self.cornerPos.y()+self.rowNum*self.cubeL)
            painter.drawText(QRect(self.cornerPos.x()+c*self.cubeL, self.cornerPos.y()+self.cubeL*self.rowNum, self.cubeL, self.cubeL),Qt.AlignCenter,str(c))

        # label
        painter.setFont(QFont('微软雅黑', 0.4*self.cubeL))
        painter.drawText(QRect(self.cornerPos.x(), self.cornerPos.y()+self.cubeL*(self.rowNum+1), self.cubeL*self.columNum, self.cubeL),Qt.AlignCenter,str('第二地点车辆数'))
        painter.rotate(90)
        painter.drawText(QRect(self.cornerPos.y(),-self.cornerPos.x()+2*self.cubeL,self.cubeL*self.columNum,-self.cubeL),Qt.AlignCenter,str('第一地点车辆数'))
