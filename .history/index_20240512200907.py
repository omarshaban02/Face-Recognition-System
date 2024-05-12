import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from ui import Ui_MainWindow
import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5.uic import loadUiType

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# from classes import Image


ui, _ = loadUiType('main.ui')


def detect_face_eyes_smiles(rgb_image):
    # Converting the image to gray
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # this returns the coordinates of the faces as x, y, w, h
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # copy of the frame
    face_only = frame.copy()
    eyes_only = frame.copy()
    face_eyes = frame.copy()
    face_smiles = frame.copy()
    face_eyes_smiles = frame.copy()

    # Looping through the faces
    for (x, y, w, h) in faces:

        # Drawing a rectangle around the face
        cv2.rectangle(face_eyes, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(face_only, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(face_smiles, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(face_eyes_smiles, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Getting the region of interest
        roi_gray = gray[y:y + h, x:x + w]
        roi_color_eyes = eyes_only[y:y + h, x:x + w]
        roi_color_smiles = face_smiles[y:y + h, x:x + w]
        roi_color_eyes_smiles = face_eyes_smiles[y:y + h, x:x + w]

        # Detecting the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 15)

        # Looping through the eyes
        for (ex, ey, ew, eh) in eyes:
            # Drawing a rectangle around the eyes
            cv2.rectangle(roi_color_eyes, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.rectangle(roi_color_eyes_smiles, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detecting the smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)

        # Looping through the smiles
        for (sx, sy, sw, sh) in smiles:
            # Drawing a rectangle around the smile
            cv2.rectangle(roi_color_smiles, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.rectangle(roi_color_eyes_smiles, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    face_only = cv2.cvtColor(face_only, cv2.COLOR_BGR2RGB)
    face_eyes = cv2.cvtColor(face_eyes, cv2.COLOR_BGR2RGB)
    face_smiles = cv2.cvtColor(face_smiles, cv2.COLOR_BGR2RGB)
    face_eyes_smiles = cv2.cvtColor(face_eyes_smiles, cv2.COLOR_BGR2RGB)

    return face_only, face_eyes, face_smiles, face_eyes_smiles


class FaceRecognizer(QMainWindow, Ui_MainWindow):
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
        self.actionOpen.triggered.connect(self.open_image)
        self.checkBox_eyes.stateChanged.connect(self.apply)
        self.checkBox_smiles.stateChanged.connect(self.apply)

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

    def load_img_file(self, image_path):

        # Loads the image using imread, converts it to RGB
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.loaded_image = image
        self.display_image(self.item_input, self.loaded_image)
        self.apply()

    def apply(self):
        # Detecting the face and eyes
        face_only, face_eyes, face_smiles, face_eyes_smiles = detect_face_eyes_smiles(self.loaded_image)

        if self.checkBox_eyes.isChecked() and self.checkBox_smiles.isChecked():
            print("Both")
            self.display_image(self.item_output, face_eyes_smiles)
        elif self.checkBox_eyes.isChecked():
            print("Eyes")
            self.display_image(self.item_output, face_eyes)
        elif self.checkBox_smiles.isChecked():
            print("Smiles")
            self.display_image(self.item_output, face_smiles)
        else:
            print("Face")
            self.display_image(self.item_output, face_only)

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        image_item.getViewBox().autoRange()

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            if plotwidget in [self.wgt_img_input, self.wgt_img_output]:
                # Removes Axes and Padding from all plotwidgets intended to display an image
                plotwidget.showAxis('left', False)
                plotwidget.showAxis('bottom', False)
                plotitem = plotwidget.getPlotItem()
                plotitem.getViewBox().setDefaultPadding(0)
            else:
                plot_title = plotwidget.objectName()[10:].capitalize()
                plotwidget.setTitle(plot_title)

            plotwidget.setBackground((25, 30, 40))
            plotwidget.showGrid(x=True, y=True)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)

    def setup_checkboxes(self):
        pass

    def init_application(self):
        self.setup_plotwidgets()
        # self.setup_checkboxes()


app = QApplication(sys.argv)
win = FaceRecognizer()
win.show()
app.exec()
