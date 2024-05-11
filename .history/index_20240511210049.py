import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from ui import Ui_MainWindow
import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5.uic import loadUiType
from classes import Image, Features, SIFT
import time

ui, _ = loadUiType('main.ui')


class FaceRecognizer(QMainWindow, ui):
    def __init__(self):
        super(FaceRecognizer, self).__init__()
        self.setupUi(self)
        
        self.loaded_image = None

        self.plotwidget_set = [self.wgt_img_input, self.wgt_img_output]

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_input, self.item_output] = [pg.ImageItem() for _ in range(2)]

        self.init_application()

        ############################### Connections ##################################################
        # Connect Openfile Action to its function
        self.actionOpen.triggered.connect(lambda: self.open_image(0))

    ##############################################################################################                
 
    ################################# Misc Functions #############################################

    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.load_img_file(selected_file)

    def load_img_file(self, image_path, target_image=0):

        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise
        image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        self.loaded_image = image
        self.display_image(self.item_input, self.loaded_image)

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.getViewBox().autoRange()

    def setup_plotwidgets(self):
        for plotwidget in [self.]:
            # Removes Axes and Padding from all plotwidgets intended to display an image
            plotwidget.showAxis('left', False)
            plotwidget.showAxis('bottom', False)
            plotwidget.setBackground((25, 30, 40))
            plotitem = plotwidget.getPlotItem()
            plotitem.getViewBox().setDefaultPadding(0)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)

    def setup_checkboxes(self):
        for checkbox in [self.lambda_chkBox, self.harris_chkBox]:
            checkbox.clicked.connect(self.corner_detection)

    def init_application(self):
        self.setup_plotwidgets()
        # self.setup_checkboxes()


app = QApplication(sys.argv)
win = FaceRecognizer()
win.show()
app.exec()
