from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from PyQt5.uic import loadUiType
from APIClient import APIClient
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog
ui, _ = loadUiType('main.ui')


class MainApp(QMainWindow, ui):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.resize(900, 1500)
        self.mouse_clicked = pyqtSignal()
        self.fileName = None
        self.kernal_size = None
        self.threshold = self.threshold_value.text()
        self.filePath1=None
        self.filePath2 =None
        # conection 
        self.open.clicked.connect(self.openImageDialog)
        self.btnSPN.clicked.connect(self.salt_paper)
        self.btnGN.clicked.connect(self.add_gaussian_noise)
        self.btnUN.clicked.connect(self.add_uniform_noise)
        # FILTER
    
        self.btnMF.clicked.connect(self.median_blur)
        self.btnAF.clicked.connect(self.blur_filter)
        self.btnGF.clicked.connect(self.gaussian_blur)
        # edge_detection
        self.boxEdge.currentIndexChanged.connect(self.handel_kernal)
        
        self.boxFreq.currentIndexChanged.connect(self.low_pass)
        # self.boxRGB.currentIndexChanged.connect(self.get_equalized_image )
        
        # thresholding buttom
        self.btnLocal.clicked.connect(self.loc_threshold)
        self.btnGlobal.clicked.connect(self.glob_threshold)
        
        # hyprid image
        self.op_photo2.clicked.connect(self.load_hybrid_2)
        self.op_photo1.clicked.connect(self.load_hybrid_1)
        self.btn_hybrid.clicked.connect(self.hyprid)
        
    
        
        
        self.apiclient = APIClient()
        self.apiclient.login("AbdullahOmran","123456789")
            
        
        

    def openImageDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.tif *.tiff);;All Files (*)",
                                                  options=options)
        if self.fileName:
            self.loadImage(self.fileName)
            

    def loadImage(self, fileName):
        pixmap = QPixmap(fileName)
        if not pixmap.isNull():
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.apiclient.upload_image(fileName) 
            scene = QGraphicsScene()
            pixmap = self.apiclient.get_grayscale()
            scene.addPixmap(pixmap)
            self.graphicsView_2.setScene(scene)
            self.graphicsView_3.setScene(scene)
            self.graphicsView_5.setScene(scene)
            self.graphicsView_8.setScene(scene)
            self.graphicsView_10.setScene(scene)
            # self.graphicsView_11.setScene(scene)
            # self.graphicsView_12.setScene(scene)
            self.graphicsView_2.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_3.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_5.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_8.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_10.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            # histograme of irignal_image
            scene = QGraphicsScene()
            pixmap = self.apiclient.get_histogram(0,'gray')
            scene.addPixmap(pixmap)
            self.graphicsView_17.setScene(scene)
            self.graphicsView_17.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            #
            # scene = QGraphicsScene()
            # pixmap = self.apiclient.get_histogram(0,'gray')
            # scene.addPixmap(pixmap)
            # self.graphicsView_16.setScene(scene)
            # self.graphicsView_16.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            # scene = QGraphicsScene()
            # pixmap = self.apiclient.get_histogram(0,'gray')
            # scene.addPixmap(pixmap)
            # self.graphicsView_16.setScene(scene)
            # self.graphicsView_16.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
    
            
    def load_hybrid_1(self): 
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.filePath1, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.tif *.tiff);;All Files (*)",
                                                  options=options)
        if self.filePath1: 
          pixmap = QPixmap(self.filePath1)
          if not pixmap.isNull():
             scene = QGraphicsScene()
             scene.addPixmap(pixmap)
             self.graphicsView_11.setScene(scene)
             self.graphicsView_11.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)  
            
    def load_hybrid_2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.filePath2, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.tif *.tiff);;All Files (*)",
                                                  options=options)
        if self.filePath2:  
          pixmap = QPixmap(self.filePath2)
          if not pixmap.isNull():
              scene = QGraphicsScene()
              scene.addPixmap(pixmap)
              self.graphicsView_12.setScene(scene)
              self.graphicsView_12.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)            
   
    def comongraphics_view_2(self,pixmap):
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_2.setScene(scene)
        self.graphicsView_2.fitInView(scene.sceneRect(), Qt.KeepAspectRatio) 
    
    def comongraphics_view_6(self,pixmap):
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_6.setScene(scene)
        self.graphicsView_6.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)            
                       
    def salt_paper(self):
        self.saltiness = self.saltiness.text()
        self.pepperiness = self.pepperiness.text()
        scene = QGraphicsScene()
        pixmap = self.apiclient.add_salt_and_pepper_noise(self.saltiness,self.pepperiness)
        self.comongraphics_view_2(pixmap)
        
        
    def add_uniform_noise(self):
        self.low = self.low_freq.text()
        self.high = self.high_freq.text()
        scene = QGraphicsScene()
        pixmap = self.apiclient.add_uniform_noise(self.low,self.high)
        self.comongraphics_view_2(pixmap)
        
    def add_gaussian_noise(self):
        self.mean_value = self.mean.text()
        self.std = self.STD.text()
        scene = QGraphicsScene()
        pixmap = self.apiclient.add_gaussian_noise(int(self.mean_value),int(self.std))
        self.comongraphics_view_2(pixmap)       
            
    def blur_filter(self):
        self.kernal_size = int(self.KS.text())
        scene = QGraphicsScene()
        pixmap = self.apiclient.blur(self.kernal_size)
        self.comongraphics_view_2(pixmap)      
    
    def gaussian_blur(self):
        
        self.STD = self.slidSTD.value()
        scene = QGraphicsScene()
        pixmap = self.apiclient.gaussian_blur(self.kernal_size,self.STD)
        self.comongraphics_view_2(pixmap)        
    
    def median_blur (self):
        scene = QGraphicsScene()
        pixmap = self.apiclient.median_blur(self.kernal_size)
        self.comongraphics_view_2(pixmap)    
    
    
    def handel_kernal(self):
       selected_kernal = self.boxEdge.currentText()
       print(selected_kernal)
       if selected_kernal == "soble":
           self.sobel_edge_detection()
           
       elif selected_kernal == "Canny":
           self.canny_edge_detection()
       elif selected_kernal == "Prewit":
           self.prewitt_edge_detection()
       elif selected_kernal == "Robarts": 
           self.roberts_edge_detection()  
                
               
               
        
    def sobel_edge_detection(self):
        scene = QGraphicsScene()
        pixmap = self.apiclient.sobel_edge_detection()
        self.comongraphics_view_6(pixmap)
        
    def prewitt_edge_detection(self):
        scene = QGraphicsScene()
        pixmap = self.apiclient.prewitt_edge_detection()
        self.comongraphics_view_6(pixmap)
        
    def roberts_edge_detection(self):
        scene = QGraphicsScene()
        pixmap = self.apiclient.roberts_edge_detection()
        self.comongraphics_view_6(pixmap)    
    #   NAQUSE GAMICA TEXT_EDITE FOR VALUE OF CANY 
    def canny_edge_detection (self):
        scene = QGraphicsScene()
        #    #   NAQUSE GAMICA TEXT_EDITE FOR VALUE OF CANY
        pixmap = self.apiclient.canny_edge_detection(50,150)
        self.comongraphics_view_6(pixmap)
    
    def get_histograme (self):
        self.channel_type = self.boxRGB.currentText()
        scene = QGraphicsScene()
        pixmap = self.apiclient.get_histogram()
        scene.addPixmap(pixmap)
        self.graphicsView_17.setScene(scene)
        self.graphicsView_17.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)  
        
    def get_equalized_image (self):
        self.channel_type = self.boxRGB.currentText()
        scene = QGraphicsScene()
        pixmap = self.apiclient.get_equalized_image()
        scene.addPixmap(pixmap)
        self.graphicsView_15.setScene(scene)
        self.graphicsView_15.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        
    def get_equalized_histograme (self):
        self.channel_type = self.boxRGB.currentText()
        scene = QGraphicsScene()
        pixmap = self.apiclient.get_equalized_histogram()
        scene.addPixmap(pixmap)
        self.graphicsView_16.setScene(scene)
        self.graphicsView_16.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)       
        
    def glob_threshold(self):
          pixmap = self.apiclient.global_threshold(self.threshold)
          scene = QGraphicsScene()
          scene.addPixmap(pixmap)
          self.graphicsView_4.setScene(scene)
          self.graphicsView_4.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)  
    def loc_threshold(self):
          pixmap = self.apiclient.local_threshold(self.threshold)
          scene = QGraphicsScene()
          scene.addPixmap(pixmap)
          self.graphicsView_4.setScene(scene)
          self.graphicsView_4.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)        
                    
        
    def hyprid(self):
        self.low_pass_cuttoff_freq = self.freq_1.value()
        self.high_pass_cuttoff_freq = self.freq2.value()
        print(self.high_pass_cuttoff_freq)
        print(self.low_pass_cuttoff_freq)
        pixmap = self.apiclient.get_hybrid_image(self.filePath1,self.filePath2,int(self.low_pass_cuttoff_freq),int(self.high_pass_cuttoff_freq))
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_7.setScene(scene)
        self.graphicsView_7.fitInView(scene.sceneRect(), Qt.KeepAspectRatio) 
        
    def low_pass(self):
        self.low_freq = self.lineFreq.text()
        pixmap = self.apiclient.get_low_pass_filter(float(self.low_freq))
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_9.setScene(scene)
        self.graphicsView_9.fitInView(scene.sceneRect(), Qt.KeepAspectRatio) 
    
            
                                
        
        
        
        
        
        
        
        
        

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
