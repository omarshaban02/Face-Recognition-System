import sys
import threading

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5.uic import loadUiType
from deepface import DeepFace
from classes import Face_Detection
from classes import Face_Recognition

ui, _ = loadUiType('main.ui')


class FaceRecognizer(QMainWindow, ui):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.loaded_image = None

        self.plotwidget_set = [self.wgt_img_input, self.wgt_img_output, self.wgt_reco_output, self.wgt_reco_input]

        self.param_scale_factor = 0
        self.param_min_neigbors = 0

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_input, self.item_output, self.item_reco_output, self.item_reco_input] \
            = [pg.ImageItem() for _ in range(4)]

        self.init_application()

        ############################### Connections ##################################################
        # Connect Openfile Action to its function
        self.actionOpen.triggered.connect(self.open_image)
        self.checkBox_eyes.stateChanged.connect(self.apply)
        self.checkBox_smiles.stateChanged.connect(self.apply)
        self.scaleFactorSlider.valueChanged.connect(self._update_scale_factor)
        self.spinBox_min_neigbors.valueChanged.connect(self._update_min_neighbors)
        self.scaleFactorSlider.setValue(8)
        self.spinBox_min_neigbors.setValue(5)
        self.btn_recognize.clicked.connect(lambda: self.recognize(eigenfaces, mean_image, transformed_data))

        # ---------------------------------- Face Recognition ----------------------------------
        centered_data, eigenfaces, mean_image = Face_Recognition.load_pca_model("./Saved-Model")
        transformed_data = Face_Recognition.transform_data(centered_data, eigenfaces)

    # ################################ Misc Functions #############################################

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
        self.display_image(self.item_reco_input, self.loaded_image)
        threading.Thread(target=self.get_analysis).start()
        self.apply()

    def apply(self):

        scale_factor = self.scaleFactorSlider.value() * 0.04 + 1
        min_neighbors = self.spinBox_min_neigbors.value()
        # Detecting the face and eyes
        face_only, face_eyes, face_smiles, face_eyes_smiles = Face_Detection.detect_face_eyes_smiles(self.loaded_image,
                                                                                                     scale_factor,
                                                                                                     min_neighbors)

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
            plotwidget.showGrid(x=False, y=False)
            plotwidget.showAxis('left', False)
            plotwidget.showAxis('bottom', False)
            plotwidget.setStyleSheet("border: 2px solid #176b87; border-radius: 5px")

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)

    def setup_checkboxes(self):
        pass

    def init_application(self):
        self.setup_plotwidgets()
        # self.setup_checkboxes()

    def _update_scale_factor(self):
        new_value = self.scaleFactorSlider.value() * 0.04 + 1
        self.label_scale_factor_value.setText(f"{new_value:.2f}")
        self.param_scale_factor = new_value
        if np.all(self.loaded_image) is not None:
            self.apply()

    def _update_min_neighbors(self):
        self.param_min_neigbors = self.spinBox_min_neigbors.value()
        if np.all(self.loaded_image) is not None:
            self.apply()

    def recognize(self, eigenfaces, mean_image, transformed_data):
        predicted_subject = Face_Recognition.predict_pca(self.loaded_image, eigenfaces, mean_image, transformed_data)
        subject_image = cv2.imread(f"./labeled_faces/s{predicted_subject:02d}_01.jpg")
        subject_image = cv2.cvtColor(subject_image, cv2.COLOR_BGR2RGB)
        self.display_image(self.item_reco_output, subject_image)

    def get_analysis(self):
        analysis = DeepFace.analyze(self.loaded_image, actions=['age', 'gender', 'emotion'])
        self.ageLabel.setText(f"{analysis[0]['age']}")
        self.genderLabel.setText(f"{analysis[0]['dominant_gender']}")
        self.emotionLabel.setText(f"{analysis[0]['dominant_emotion']}")


app = QApplication(sys.argv)
win = FaceRecognizer()
win.show()
app.exec()
