import sys
from PyQt5.Qt import *
from PyQt5 import QtGui

class GUI(QWidget):

    def __init__(self):
        super().__init__()
        self.window()

    # 窗口
    def window(self):
        self.resize(800,600) # 窗口大小
        self.setWindowTitle("数字识别") # 窗口名称
        self.setWindowIcon(QIcon("123.png")) # 程序图标

        self.display_center()

        self.picture_box()

        self.path_box()

        self.button()

        self.show() # 显示窗口

    # 图片显示框
    def picture_box(self):
        self.label = QLabel(self)
        self.label.setFixedSize(350, 300)
        self.label.move(200, 200)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;}"
                                 )

    # 路径框
    def path_box(self):
        self.qle = QLineEdit(self)
        self.qle.setGeometry(200,50,300,20)

    # 中心显示窗口
    def display_center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 创建按钮
    def button(self):
        self.btn1 = QPushButton("选择图片", self)
        self.btn1.resize(self.btn1.sizeHint())  # 创建按键
        self.btn1.move(200,100)  # 按键位置
        self.btn1.clicked.connect(self.openimage) # 点击按键发生事件

        self.btn2 = QPushButton("上传", self)
        self.btn2.resize(self.btn2.sizeHint())
        self.btn2.move(300,100)
        self.btn2.clicked.connect(self.upload)

    # 选择事件
    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self,"选择图片","/")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.path = imgName

    # 上传事件
    def upload(self):
        self.qle.insert(self.path)# 插入到路径框

    # 退出提示
    def closeEvent(self, event):

        reply = QMessageBox.question(self,
                                     '退出',
                                     "Are you sure to quit?",
                                     QMessageBox.No | QMessageBox.Yes,
                                     QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())