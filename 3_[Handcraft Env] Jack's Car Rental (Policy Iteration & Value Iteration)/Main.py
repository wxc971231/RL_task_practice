# 编译.ui文件：pyuic5 ./UI/xxxx.ui -o ./UI/xxxx.py 
import sys
from PyQt5.QtWidgets import QApplication
from Enviroment import EnviromentUI
from UI import MainWindowUI
from Enviroment import Enviroment

if __name__ == '__main__':
    app = QApplication(sys.argv)   
    env = Enviroment()
    MainWindow = MainWindowUI(env)
    MainWindow.show()
    sys.exit(app.exec_())    
