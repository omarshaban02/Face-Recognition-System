# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1105, 816)
        MainWindow.setStyleSheet("QWidget {\n"
"  background-color: #19232D;\n"
"  border: 0px solid #455364;\n"
"  padding: 0px;\n"
"  color: #E0E1E3;\n"
"  selection-background-color: #346792;\n"
"  selection-color: #E0E1E3;\n"
"    font-family : Ubuntu;\n"
"font-size: 20px;\n"
"font-weight: bold;\n"
"}\n"
"\n"
"QWidget:disabled {\n"
"  background-color: #19232D;\n"
"  color: #9DA9B5;\n"
"  selection-background-color: #26486B;\n"
"  selection-color: #9DA9B5;\n"
"}\n"
"\n"
"QWidget::item:selected {\n"
"  background-color: #176B87;\n"
"}\n"
"\n"
"QWidget::item:hover:!selected {\n"
"  background-color: rgba(23, 107, 135,50%);\n"
"}\n"
"\n"
"QMainWindow::separator {\n"
"  background-color: #455364;\n"
"  border: 0px solid #19232D;\n"
"  spacing: 0px;\n"
"  padding: 2px;\n"
"}\n"
"\n"
"QMainWindow::separator:hover {\n"
"  background-color: #60798B;\n"
"  border: 0px solid #1A72BB;\n"
"}\n"
"\n"
"/*-------------------------------------------------------------------------------------------*/\n"
"\n"
"QCheckBox {\n"
"  background-color: #19232D;\n"
"  color: #E0E1E3;\n"
"  spacing: 4px;\n"
"  outline: none;\n"
"  padding-top: 4px;\n"
"  padding-bottom: 4px;\n"
"}\n"
"\n"
"\n"
"\n"
"QCheckBox QWidget:disabled {\n"
"  background-color: #19232D;\n"
"  color: #9DA9B5;\n"
"}\n"
"\n"
"QCheckBox::indicator {\n"
"  margin-left: 2px;\n"
"  height: 14px;\n"
"  width: 14px;\n"
"}\n"
"\n"
"/*-------------------------------------------------------------------------------------------*/\n"
"QScrollBar:horizontal {\n"
"  height: 16px;\n"
"  margin: 2px 16px 2px 16px;\n"
"  border: 1px solid white;\n"
"  border-radius: 4px;\n"
"  background-color: rgb(25, 35, 45);\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal {\n"
"  background-color: #176B87;\n"
"  border: 1px solid #C9CDD0;\n"
"  border-radius: 4px;\n"
"  min-width: 8px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal:hover {\n"
"  background-color: rgba(23, 107, 135,80%);\n"
"  border:#4169E1;\n"
"  border-radius: 4px;\n"
"  min-width: 8px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal:focus {\n"
"  border: 1px solid red;\n"
"}\n"
"QScrollBar:vertical {\n"
"  background-color: rgb(25, 35, 45);\n"
"  width: 16px;\n"
"  margin: 16px 2px 16px 2px;\n"
"  border: 1px solid white;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"\n"
"QScrollBar::handle:vertical {\n"
"  background-color: #176B87;\n"
"  border: 1px solid #C9CDD0;\n"
"  min-height: 8px;\n"
"  border-radius: 4px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical:hover {\n"
"  background-color: rgba(23, 107, 135,80%);\n"
"  border: #9FCBFF;\n"
"  border-radius: 4px;\n"
"  min-height: 8px;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical:focus {\n"
"  border: 1px solid #73C7FF;\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-line:horizontal {\n"
"  margin: 0px 0px 0px 0px;\n"
"  border-image: url(\"qss_icons/light/rc/arrow_right_disabled.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: right;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line:horizontal:hover, QScrollBar::add-line:horizontal:on {\n"
"  border-image: url(\"qss_icons/light/rc/arrow_right.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: right;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical {\n"
"  margin: 3px 0px 3px 0px;\n"
"  border-image: url(\"qss_icons/light/rc/arrow_down_disabled.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: bottom;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical:hover, QScrollBar::add-line:vertical:on {\n"
"  border-image: url(\"qss_icons/light/rc/arrow_down.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: bottom;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal {\n"
"  margin: 0px 3px 0px 3px;\n"
"  border-image: url(\"qss_icons/light/rc/arrow_left_disabled.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: left;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:horizontal:hover, QScrollBar::sub-line:horizontal:on {\n"
"  border-image: url(\"qss_icons/light/rc/arrow_left.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: left;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical {\n"
"  margin: 3px 0px 3px 0px;\n"
"  border-image: url(\"qss_icons/light/rc/arrow_up_disabled.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: top;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::sub-line:vertical:hover, QScrollBar::sub-line:vertical:on {\n"
"  border-image: url(\"qss_icons/light/rc/arrow_up.png\");\n"
"  height: 12px;\n"
"  width: 12px;\n"
"  subcontrol-position: top;\n"
"  subcontrol-origin: margin;\n"
"}\n"
"\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal {\n"
"  background: none;\n"
"}\n"
"\n"
"QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"  background: none;\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {\n"
"  background: none;\n"
"}\n"
"\n"
"QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"  background: none;\n"
"}\n"
"\n"
"/*----------------------------------------------------------------------------------------*/\n"
"QPushButton{\n"
" /*border:2px solid #05B8CC;*/\n"
" background-color: #176B87;\n"
" color:rgb(255, 255, 255);\n"
" border-radius: 10px;\n"
" font-weight:bold;\n"
" font-size:16px;\n"
"  transition: 500ms;\n"
"}\n"
"\n"
"QPushButton::hover{\n"
" border: 1px solid #176B87;\n"
" background-color: rgba(23, 107, 135,80%)\n"
"}\n"
"QPushButton:pressed {\n"
" margin:1px 2px;\n"
" font-size: 15px;\n"
"}\n"
"/*----------------------------------------------------------------------------------------*/\n"
"QTabBar::tab {\n"
"    background-color: #455364;\n"
"    border: 3px solid #19232D;\n"
"    border-radius: 10px;\n"
"    padding: 5px;\n"
"    color: #E0E1E3;\n"
"    font-family: Ubuntu;\n"
"    font-size: 20px;\n"
"    font-weight: bold;\n"
"}\n"
"QTabBar::tab {\n"
"    background-color: #455364;\n"
"    border: 3px solid #19232D;\n"
"    border-radius: 10px;\n"
"    padding: 4px 8px;\n"
"    color: #E0E1E3;\n"
"    font-family: Ubuntu;\n"
"    font-size: 20px;\n"
"    font-weight: bold;\n"
"}\n"
"QTabBar::tab:selected {\n"
"    background-color:rgba(69,83,100,60%);\n"
"    color: white;\n"
"    margin-top: 3px;\n"
"    border-bottom: 3px solid white;\n"
"}\n"
"QPlotWidget{\n"
" background-color:(25, 35, 45, 0.8)\n"
"}\n"
"\n"
"\n"
"QCheckBox{\n"
"    font-size: 18px;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setContentsMargins(11, 5, -1, 5)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(11)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox_faces = QtWidgets.QCheckBox(self.tab)
        self.checkBox_faces.setObjectName("checkBox_faces")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.checkBox_faces)
        self.horizontalLayout_2.addWidget(self.checkBox_faces)
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab)
        self.checkBox_2.setObjectName("checkBox_2")
        self.buttonGroup.addButton(self.checkBox_2)
        self.horizontalLayout_2.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(self.tab)
        self.checkBox_3.setObjectName("checkBox_3")
        self.buttonGroup.addButton(self.checkBox_3)
        self.horizontalLayout_2.addWidget(self.checkBox_3)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setObjectName("label")
        self.horizontalLayout_7.addWidget(self.label)
        self.horizontalSlider = QtWidgets.QSlider(self.tab)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_7.addWidget(self.horizontalSlider)
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setMinimumSize(QtCore.QSize(60, 0))
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_7.addWidget(self.label_9)
        self.horizontalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_8.addWidget(self.label_2)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.tab)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalLayout_8.addWidget(self.horizontalSlider_2)
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setMinimumSize(QtCore.QSize(60, 0))
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_8.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.horizontalLayout_8)
        self.btn_apply = QtWidgets.QPushButton(self.tab)
        self.btn_apply.setMinimumSize(QtCore.QSize(100, 0))
        self.btn_apply.setObjectName("btn_apply")
        self.horizontalLayout.addWidget(self.btn_apply)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.wgt_img_input = PlotWidget(self.tab)
        self.wgt_img_input.setStyleSheet("background:beige;")
        self.wgt_img_input.setObjectName("wgt_img_input")
        self.verticalLayout.addWidget(self.wgt_img_input)
        self.wgt_graph_performance = PlotWidget(self.tab)
        self.wgt_graph_performance.setStyleSheet("background: orange;")
        self.wgt_graph_performance.setObjectName("wgt_graph_performance")
        self.verticalLayout.addWidget(self.wgt_graph_performance)
        self.wgt_graph_ROC = PlotWidget(self.tab)
        self.wgt_graph_ROC.setStyleSheet("background: dark blue;")
        self.wgt_graph_ROC.setObjectName("wgt_graph_ROC")
        self.verticalLayout.addWidget(self.wgt_graph_ROC)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.wgt_img_output = PlotWidget(self.tab)
        self.wgt_img_output.setStyleSheet("background: crimson")
        self.wgt_img_output.setObjectName("wgt_img_output")
        self.gridLayout.addWidget(self.wgt_img_output, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 2)
        self.gridLayout.setRowStretch(1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1105, 30))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.valueChanged['int'].connect(self.label_9.setNum) # type: ignore
        self.horizontalSlider_2.valueChanged['int'].connect(self.label_5.setNum) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox_faces.setText(_translate("MainWindow", "Faces"))
        self.checkBox_2.setText(_translate("MainWindow", "Smiles"))
        self.checkBox_3.setText(_translate("MainWindow", "Eyes"))
        self.label.setText(_translate("MainWindow", "TBA"))
        self.label_9.setText(_translate("MainWindow", "0"))
        self.label_2.setText(_translate("MainWindow", "TBA"))
        self.label_5.setText(_translate("MainWindow", "0"))
        self.btn_apply.setText(_translate("MainWindow", "Apply"))
        self.wgt_img_input.setWhatsThis(_translate("MainWindow", "Widget Holding the opened image"))
        self.wgt_graph_performance.setWhatsThis(_translate("MainWindow", "Widget Holding the opened image"))
        self.wgt_graph_ROC.setWhatsThis(_translate("MainWindow", "Widget holding the edge detection result for colored image"))
        self.wgt_img_output.setWhatsThis(_translate("MainWindow", "Widget holding the edge detection result for colored image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Face Detection"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
