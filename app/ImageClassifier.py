import os
from sys import exit, argv
import xml.etree.ElementTree as ET
from time import sleep

import numpy as np
from PyQt5 import uic, Qt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPixmap
from PyQt5.QtOpenGL import QGLFormat
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QMessageBox, QAction, QFileDialog
from PIL import Image
from threading import Thread

current_folder = os.getcwd().replace('\\', '/')
model_filename = 'C_CT_S.h5'


class ProcessingThread(QThread):
    model_loaded = pyqtSignal()
    classification_done = pyqtSignal(np.ndarray)

    def __init__(self):
        QThread.__init__(self)
        self.image = self.ml_model = None
        Thread(target=self.load_model).start()

    def load_model(self):
        from modules.cancerClassifier.trainApp import create_model
        size = (224, 224, 3)
        model = create_model(size)
        model_filepath = current_folder + '/modules/cancerClassifier/models/' + model_filename
        model.load_weights(model_filepath)
        print('model ready')
        self.model_loaded.emit()
        self.ml_model = model
        # from keras.models import load_model
        # model_filepath = current_folder + '/modules/cancerClassifier/models/' + model_filename
        # model = load_model(model_filepath)

    def run(self):
        try:
            res = self.ml_model.predict(self.image)
            self.classification_done.emit(res[0])
        except Exception as e:
            print(e)

    def load_image(self, img_path):
        while not self.ml_model:
            sleep(0.1)
        from keras.utils import load_img, img_to_array
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        self.image = img_array / 255.0


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.image_scene = QGraphicsScene()
        self.proc_th = ProcessingThread()
        self.image = None
        uic.loadUi("res/main_window.ui", self)
        self.init_controls()
        self.show()
        self.class_list = []
        self.model = None
        self.modules_path = current_folder + '/modules/'
        self.classifier_type = self.get_classifier_type()
        self.load_classifier_description()

    def get_classifier_type(self):
        items = os.listdir(self.modules_path)
        directories = [item for item in items if os.path.isdir(os.path.join(self.modules_path, item))]
        return directories[0]

    def load_classifier_description(self):
        descr_path = self.modules_path + self.classifier_type + '/description.xml'
        tree = ET.parse(descr_path)
        root = tree.getroot()
        classifier_type = root.attrib['type']
        # Заповнення масиву значеннями з XML елементів
        for class_element in root.findall('class'):
            class_name = class_element.attrib['name']
            class_description = class_element.attrib['description']
            self.class_list.append([class_name, class_description])
        self.setWindowTitle(f"{classifier_type} classifier")

    def load_image(self):
        image_path = \
            QFileDialog.getOpenFileName(self, "Select an image", filter="Image(*.jpg *.png *.bmp *.jpeg)")[0]
        # image_path = "D:/Diploma/app/modules/cancerClassifier/dataset/test/adenocarcinoma/000108 (3).png"
        if image_path:
            self.image = np.asarray(Image.open(image_path).convert('RGB'))
            image_height, image_width, _ = self.image.shape
            q_image = QImage(self.image.data, image_width, image_height, self.image.strides[0],
                             QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_scene.addPixmap(pixmap)
            self.imagePathTxt.setText(image_path)
            self.imageView.scene().setSceneRect(0, 0, image_width, image_height)
            self.imageView.setMinimumWidth(image_width)
            self.imageView.setMinimumHeight(image_height)
            self.classifierBox.setMaximumWidth(16777215)
            self.classifierBox.setMaximumHeight(16777215)
            self.statusbar.showMessage(f"Opened image "
                                       f"'{image_path.removeprefix(image_path[:image_path.rfind('/') + 1])}'")
            Thread(target=lambda: self.proc_th.load_image(image_path)).start()

    def classify_image(self):
        self.proc_th.start()
        self.classifyImageButton.setEnabled(False)
        self.classifyImageButton.setText("Classification...")

    def init_controls(self):
        # recognize_act = QAction("Розпізнати образ", self)
        # self.menubar.addAction(recognize_act)
        self.imageView.setScene(self.image_scene)
        self.selectFile.clicked.connect(self.load_image)
        self.classifyImageButton.clicked.connect(self.classify_image)
        # recognize_act.triggered.connect(self.recognize_image)

        self.imageView.setCacheMode(self.imageView.CacheBackground)
        self.imageView.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        if QGLFormat.hasOpenGL():
            self.imageView.setRenderHint(QPainter.HighQualityAntialiasing)
        self.imageView.setViewportUpdateMode(self.imageView.SmartViewportUpdate)
        self.imageView.setHorizontalScrollBarPolicy(Qt.Qt.ScrollBarAlwaysOff)
        self.imageView.setVerticalScrollBarPolicy(Qt.Qt.ScrollBarAlwaysOff)

        self.proc_th.model_loaded.connect(self.model_ready)
        self.proc_th.classification_done.connect(self.show_classification_res)

    def show_classification_res(self, res):
        res_text = "<center><b>" + "Class".rjust(35) + "  |  proba</b><br>" + ("-" * 35) + "<br>"
        for i, class_name in enumerate(self.class_list):
            res_text += class_name[0].rjust(25) + "  |  " + str(round(res[i], 2)) + "<br>"
        probable_class_ind = np.argmax(res)
        probable_class = self.class_list[probable_class_ind][0]
        res_text = res_text.replace(' ', "&nbsp;") + f"</center>. <br><b>Conclusion: {probable_class}.</b>"
        self.classifyImageButton.setEnabled(True)
        self.classifyImageButton.setText("Classify image")
        msg = QMessageBox(QMessageBox.Information, f"Classification result", res_text)
        msg.exec()

    def model_ready(self):
        self.statusbar.showMessage(f"Classification model loaded!")
        self.classifyImageButton.setEnabled(True)
        self.classifyImageButton.setText("Classify image")

    def resizeEvent(self, event):
        rect = self.imageView.scene().itemsBoundingRect()
        self.imageView.fitInView(rect, Qt.Qt.KeepAspectRatio)
        self.imageView.fitInView(rect, Qt.Qt.KeepAspectRatio)

    def closeEvent(self, event):
        self.proc_th.terminate()
        self.close()


if __name__ == '__main__':
    app = QApplication(argv)
    ex = MainWindow()
    exit(app.exec_())
