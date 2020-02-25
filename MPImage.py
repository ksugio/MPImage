
import MPImfp
import MPLn23d
import cv2
import sys
import random
import numpy as np
import json
import math as m
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

"""
Image Scene
"""  
class ImageScene(QtWidgets.QGraphicsScene):
  addLine = QtCore.pyqtSignal(float, float, float, float)
  addRect = QtCore.pyqtSignal(float, float, float, float)
  clearRect = QtCore.pyqtSignal(float, float, float, float)
  measurePixel = QtCore.pyqtSignal(float)
  mouseNone = 0
  mouseLine = 1
  mouseRect = 2
  mouseClear = 3
  mouseMeasure = 4

  def __init__(self, *argv, **keywords):
    super(ImageScene, self).__init__(*argv, **keywords)
    self.cvimg = None
    self.imageItem = None
    self.roiItem = None
    self.scale = 1.0
    self.startPos = None
    self.mouseMode = self.mouseNone

  def setImage(self, cvimg):
    self.cvimg = cvimg
    if len(cvimg.shape) == 2:
      height, width = cvimg.shape
      qimg = QtGui.QImage(cvimg.data, width, height, width, QtGui.QImage.Format_Indexed8)
    elif len(cvimg.shape) == 3:  
      height, width, dim = cvimg.shape      
      qimg = QtGui.QImage(cvimg.data, width, height, dim * width, QtGui.QImage.Format_RGB888)
      qimg = qimg.rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(qimg)
    if self.imageItem:
      self.removeItem(self.imageItem)
    self.imageItem = QtWidgets.QGraphicsPixmapItem(pixmap)
    self.addItem(self.imageItem)
    self.__Scale()

  def clearImage(self):
    self.cvimg = None
    if self.imageItem:
      self.removeItem(self.imageItem)
      self.imageItem = None

  def __Scale(self):
    if self.imageItem:
      self.imageItem.setScale(self.scale)
      w = self.scale * self.cvimg.shape[1]
      h = self.scale * self.cvimg.shape[0]
      self.setSceneRect(0, 0, w, h)
    if self.roiItem:
      self.roiItem.setScale(self.scale)

  def setScale(self, scale):
    self.scale = scale
    self.__Scale()

  def calcFitScale(self):
    if self.cvimg is not None:
      view = self.views()[0]      
      sw = float(view.width()) / float(self.cvimg.shape[1])
      sh = float(view.height()) / float(self.cvimg.shape[0])
      if sw < sh:
        return sw
      else:
        return sh
    else:
      return 1.0

  def drawROI(self, x, y, w, h):
    if self.roiItem:
      self.removeItem(self.roiItem)    
    self.roiItem = QtWidgets.QGraphicsRectItem()
    pen = QtGui.QPen(QtGui.QColor(255,0,0))
    self.roiItem.setPen(pen)
    self.roiItem.setRect(x, y, w, h)
    self.roiItem.setScale(self.scale)
    self.addItem(self.roiItem)

  def clearROI(self):
    if self.roiItem:
      self.removeItem(self.roiItem)
      self.roiItem = None
      
  def clearAll(self):
    self.clearImage()
    self.clearROI()

  def mousePressEvent(self, event):
    if self.imageItem and event.button() == QtCore.Qt.LeftButton:
      self.startPos = event.scenePos()
      if self.mouseMode == self.mouseLine:
        self.line_item = QtWidgets.QGraphicsLineItem()
        pen = QtGui.QPen(QtGui.QColor(0,255,0))
        self.line_item.setPen(pen)
        self.addItem(self.line_item)
      elif self.mouseMode == self.mouseRect:
        self.rect_item = QtWidgets.QGraphicsRectItem()
        pen = QtGui.QPen(QtGui.QColor(0,255,0))
        self.rect_item.setPen(pen)
        self.addItem(self.rect_item)
      elif self.mouseMode == self.mouseClear:
        self.rect_item = QtWidgets.QGraphicsRectItem()
        pen = QtGui.QPen(QtGui.QColor(0,0,255))
        self.rect_item.setPen(pen)
        self.addItem(self.rect_item)        
      elif self.mouseMode == self.mouseMeasure:
        self.line_item = QtWidgets.QGraphicsLineItem()
        pen = QtGui.QPen(QtGui.QColor(255,0,0))
        self.line_item.setPen(pen)
        self.addItem(self.line_item)

  def mouseMoveEvent(self, event):
    if self.startPos:
      start = self.startPos
      cur = event.scenePos()
      if self.mouseMode == self.mouseLine:
        self.line_item.setLine(start.x(), start.y(), cur.x(), cur.y())
      elif self.mouseMode == self.mouseRect:  
        self.rect_item.setRect(start.x(), start.y(), cur.x() - start.x(), cur.y() - start.y())
      elif self.mouseMode == self.mouseClear:  
        self.rect_item.setRect(start.x(), start.y(), cur.x() - start.x(), cur.y() - start.y())        
      elif self.mouseMode == self.mouseMeasure:
        self.line_item.setLine(start.x(), start.y(), cur.x(), cur.y())        

  def mouseReleaseEvent(self, event):
    if self.startPos:
      sx = self.startPos.x() / self.scale
      sy = self.startPos.y() / self.scale
      ex = event.scenePos().x() / self.scale
      ey = event.scenePos().y() / self.scale
      if self.mouseMode == self.mouseLine:
        self.removeItem(self.line_item)
        self.addLine.emit(sx, sy, ex, ey)
      elif self.mouseMode == self.mouseRect:
        self.removeItem(self.rect_item)
        self.addRect.emit(sx, sy, ex, ey)
      elif self.mouseMode == self.mouseClear:
        self.removeItem(self.rect_item)
        self.clearRect.emit(sx, sy, ex, ey)
      elif self.mouseMode == self.mouseMeasure:
        self.removeItem(self.line_item)
        dx = ex - sx
        dy = ey - sy
        dis = m.sqrt(dx * dx + dy * dy)
        self.measurePixel.emit(dis)
      self.startPos = None
    super(ImageScene, self).mouseReleaseEvent(event)

  def setMouseMode(self, mode):
    self.mouseMode = mode

"""
Graphics View
"""  
class GraphicsView(QtWidgets.QGraphicsView):
  def __init__(self):
    super(GraphicsView, self).__init__()
    self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
    self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform | QtGui.QPainter.TextAntialiasing)

  def sizeHint(self):
    return QtCore.QSize(1024, 768)

"""
File Widget
"""  
class FileWidget(QtWidgets.QWidget):
  imageChanged = QtCore.pyqtSignal(np.ndarray)
  roiChanged = QtCore.pyqtSignal(int, int, int, int)
  Units = ['px', 'km', 'm', 'cm', 'mm', 'um', 'nm']
  
  def __init__(self, parent):
    super(FileWidget, self).__init__(parent)
    self.org_img = None
    self.insitu = True
    vbox = QtWidgets.QVBoxLayout(self)
    self.name_label = QtWidgets.QLabel('Name :')
    vbox.addWidget(self.name_label)
    self.size_label = QtWidgets.QLabel('Size :')
    vbox.addWidget(self.size_label)
    vbox.addWidget(QtWidgets.QLabel('Resize :'))
    hbox = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox)
    self.spin5 = QtWidgets.QSpinBox()
    self.spin5.valueChanged[int].connect(self.changeWidth)
    hbox.addWidget(self.spin5)
    self.spin6 = QtWidgets.QSpinBox()
    self.spin6.valueChanged[int].connect(self.changeHeight)
    hbox.addWidget(self.spin6)    
    vbox.addWidget(QtWidgets.QLabel('ROI :'))
    glay = QtWidgets.QGridLayout()
    vbox.addLayout(glay)
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin1, 0, 0)
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin2, 0, 1)
    self.spin3 = QtWidgets.QSpinBox()
    self.spin3.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin3, 1, 0)
    self.spin4 = QtWidgets.QSpinBox()
    self.spin4.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin4, 1, 1)
    button2 = QtWidgets.QPushButton('Reset ROI')
    button2.clicked[bool].connect(self.resetROI)
    vbox.addWidget(button2)
    vbox.addWidget(QtWidgets.QLabel('Pixels in Scale :'))
    hbox0 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox0)
    self.line1 = QtWidgets.QLineEdit()
    self.line1.setValidator(QtGui.QDoubleValidator())
    self.line1.setText('1.0')
    hbox0.addWidget(self.line1)
    self.button3 = QtWidgets.QPushButton('Measure')
    self.button3.setCheckable(True)
    hbox0.addWidget(self.button3)
    vbox.addWidget(QtWidgets.QLabel('Scale :'))
    hbox1 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.line2 = QtWidgets.QLineEdit()
    self.line2.setValidator(QtGui.QDoubleValidator())
    self.line2.setText('1.0')
    hbox1.addWidget(self.line2)
    self.combo = QtWidgets.QComboBox()
    for unit in self.Units:
      self.combo.addItem(unit)
    hbox1.addWidget(self.combo)
    self.check1 = QtWidgets.QCheckBox('Measure')
    self.check1.setChecked(True)
    vbox.addWidget(self.check1)
    vbox.addStretch()

  def clearFile(self):
    self.insitu = False
    self.org_img = None
    self.name_label.setText('Name :')
    self.size_label.setText('Size :')
    self.spin1.setValue(0)
    self.spin2.setValue(0)        
    self.spin3.setValue(0)
    self.spin4.setValue(0)
    self.line1.setText('1.0')
    self.line2.setText('1.0')
    self.combo.setCurrentIndex(0)
    self.check1.setChecked(True)
    self.insitu = True
    
  def setInfo(self, info):
    self.insitu = False
    roi = info['FileROI']
    size = info['FileSize']
    resize = info['FileResize']
    fname = info['FileName']
    filename = QtCore.QFileInfo(fname).fileName()
    self.name_label.setText('Name : ' + filename)    
    self.size_label.setText('Size : ' + str(size[0]) + ' x ' + str(size[1]))
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(resize[0])
    self.spin1.setValue(roi[0])
    self.spin2.setMinimum(0)
    self.spin2.setMaximum(resize[1])   
    self.spin2.setValue(roi[1])
    self.spin3.setMinimum(0)    
    self.spin3.setMaximum(resize[0])
    self.spin3.setValue(roi[2])
    self.spin4.setMinimum(0)
    self.spin4.setMaximum(resize[1]) 
    self.spin4.setValue(roi[3])
    self.spin5.setMinimum(1)
    self.spin5.setMaximum(10 * size[0]) 
    self.spin5.setValue(resize[0])
    self.spin6.setMinimum(1)
    self.spin6.setMaximum(10 * size[1])
    self.spin6.setValue(resize[1])
    self.insitu = True
    self.line1.setText(str(info['FilePixels']))
    self.line2.setText(str(info['FileScale']))
    self.combo.setCurrentIndex(info['FileUnit'])
    self.check1.setChecked(info['FileMeasure'])
    
  def getInfo(self, fname=None):
    info = {}
    if fname == None:
      if self.org_img is not None:
        height, width, dim = self.org_img.shape
        info['FileSize'] = [width, height]
      else:
        info['FileSize'] = [0, 0]
      info['FileResize'] = [self.spin5.value(), self.spin6.value()]
      info['FileROI'] = [self.spin1.value(), self.spin2.value(), self.spin3.value(), self.spin4.value()]
    else:
      img = cv2.imread(str(fname), 1)
      height, width, dim = img.shape
      info['FileSize'] = [width, height]
      info['FileResize'] = [width, height]
      info['FileROI'] = [0, 0, width, height]
    info['FilePixels'] = float(self.line1.text())
    info['FileScale'] = float(self.line2.text())
    info['FileUnit'] = self.combo.currentIndex()
    info['FileMeasure'] = self.check1.isChecked()
    return info

  def changeROI(self):
    if self.insitu:
      sx = self.spin1.value()
      sy = self.spin2.value()
      ex = self.spin3.value()
      ey = self.spin4.value()
      self.roiChanged.emit(sx, sy, ex - sx, ey - sy)
 
  def resetROI(self):
    self.initROI()
    self.changeROI()

  def initROI(self):
    width = self.spin5.value()
    height = self.spin6.value()
    self.insitu = False
    self.spin1.setMaximum(width)
    self.spin1.setValue(0)    
    self.spin2.setMaximum(height)   
    self.spin2.setValue(0)    
    self.spin3.setMaximum(width)
    self.spin3.setValue(width)    
    self.spin4.setMaximum(height) 
    self.spin4.setValue(height)
    self.insitu = True

  def changeWidth(self):
    if self.insitu:
      height, width, dim = self.org_img.shape
      self.insitu = False
      self.spin6.setValue(self.spin5.value() * height / width)
      self.insitu = True
      self.initROI()
      self.setImage()

  def changeHeight(self):
    if self.insitu:
      height, width, dim = self.org_img.shape
      self.insitu = False
      self.spin5.setValue(self.spin6.value() * width / height)
      self.insitu = True
      self.initROI()
      self.setImage()

  def setImage(self):
    if self.org_img is not None:
      height, width, dim = self.org_img.shape
      if self.spin5.value() == width and self.spin6.value() == height:
        self.imageChanged.emit(self.org_img)
      else:
        res_img = cv2.resize(self.org_img, (self.spin5.value(), self.spin6.value()))
        self.imageChanged.emit(res_img)
      sx = self.spin1.value()
      sy = self.spin2.value()
      ex = self.spin3.value()
      ey = self.spin4.value()
      self.roiChanged.emit(sx, sy, ex - sx, ey - sy)

  def measurePixel(self, dis):
    if self.button3.isChecked():
      mdis = dis * self.org_img.shape[1] / self.spin5.value()
      self.line1.setText(str(mdis))

  @staticmethod
  def Process(src_img, info):
    if src_img is not None:
      height, width, dim = src_img.shape
      roi = info['FileROI']
      resize = info['FileResize']
      if resize[0] == width and resize[1] == height:
        return src_img[roi[1]:roi[3], roi[0]:roi[2]]
      else:
        res_img = cv2.resize(src_img, (resize[0], resize[1]))
        return res_img[roi[1]:roi[3], roi[0]:roi[2]]
    else:
      return None

  @staticmethod
  def PixelSize(info):
    size = info['FileSize']
    resize = info['FileResize']
    pixels = info['FilePixels']
    scale = info['FileScale']
    unit = info['FileUnit']
    ps = scale / pixels * size[0] / resize[0]
    return [ps, FileWidget.Units[unit]]

"""
Contrast Widget
"""  
class ContrastWidget(QtWidgets.QWidget):
  imageChanged = QtCore.pyqtSignal(np.ndarray)
  
  def __init__(self, parent):
    super(ContrastWidget, self).__init__(parent)
    self.src_img = None
    self.insitu = True
    vbox = QtWidgets.QVBoxLayout(self)
    self.figure = Figure(figsize=(1,2))
    self.figure.subplots_adjust(bottom=0.1, top=0.98, left=0.02, right=0.98, hspace=0.02)
    self.canvas = FigureCanvas(self.figure)
    vbox.addWidget(self.canvas)
    self.check1 = QtWidgets.QCheckBox('Background Subtraction')
    self.check1.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check1)
    vbox.addWidget(QtWidgets.QLabel('Filter Size :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(1999)
    self.spin1.setSingleStep(2)
    self.spin1.setValue(255)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('LUT Min. and Max. :'))
    hbox = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox)
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(0)
    self.spin2.setMaximum(255)
    self.spin2.setValue(0)
    self.spin2.valueChanged[int].connect(self.setImage)
    hbox.addWidget(self.spin2)
    self.spin3 = QtWidgets.QSpinBox()
    self.spin3.setMinimum(0)
    self.spin3.setMaximum(255)
    self.spin3.setValue(255)
    self.spin3.valueChanged[int].connect(self.setImage)
    hbox.addWidget(self.spin3)
    vbox.addStretch()

  def setInfo(self, info):
    self.insitu = False
    self.check1.setChecked(info['ContBGSubtract'])
    self.spin1.setValue(info['ContFilterSize'])
    self.spin2.setValue(info['ContLUTMin'])
    self.spin3.setValue(info['ContLUTMax'])
    self.insitu = True

  def getInfo(self):
    info = {}
    info['ContBGSubtract'] = self.check1.isChecked()
    info['ContFilterSize'] = self.spin1.value()
    info['ContLUTMin'] = self.spin2.value()
    info['ContLUTMax'] = self.spin3.value()
    return info

  def setImage(self):
    if self.insitu:
      dst_img = self.Process(self.src_img, self.getInfo())
      if dst_img is not None:
        self.drawGraph(self.src_img, dst_img)
        self.imageChanged.emit(dst_img)

  def drawHist(self, img, ax):
    lv = range(256)
    if len(img.shape) == 2:
      hist = cv2.calcHist([img], [0], None, [256], [0,256])
      ax.plot(lv, hist, 'k')
    elif len(img.shape) == 3:
      col = ['b', 'g', 'r']
      for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        ax.plot(lv, hist, col[i])
    ax.set_xlim(0, 255)

  def drawGraph(self, src_img, dst_img):
    self.figure.clf()
    ax1 = self.figure.add_subplot(2,1,1)
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    self.drawHist(src_img, ax1)
    ax2 = self.figure.add_subplot(2,1,2)
    ax2.yaxis.set_ticklabels([])
    self.drawHist(dst_img, ax2)
    self.canvas.draw()

  @staticmethod
  def Process(src_img, info):
    if src_img is None:
      return None
    if info['ContBGSubtract']:
      fsize = info['ContFilterSize']
      fore = src_img.astype(np.int32)
      back = cv2.blur(src_img, (fsize, fsize)).astype(np.int32)
      sub = fore - back + 127
      img = sub.astype(np.uint8)
    else:
      img = src_img
    lutmin = info['ContLUTMin']
    lutmax = info['ContLUTMax']
    diff = lutmax - lutmin
    lut = np.empty((256, 1), dtype='uint8')
    for i in range(256):
      if i <= lutmin:
        lut[i][0] = 0
      elif i >= lutmax:
        lut[i][0] = 255
      else:
        lut[i][0] = 255 * (i - lutmin) / diff
    return cv2.LUT(img, lut)

"""
Filter Widget
"""  
class FilterWidget(QtWidgets.QWidget):
  imageChanged = QtCore.pyqtSignal(np.ndarray)
  
  def __init__(self, parent):
    super(FilterWidget, self).__init__(parent)
    self.src_img = None
    self.insitu = True
    vbox = QtWidgets.QVBoxLayout(self)
    vbox.addWidget(QtWidgets.QLabel('Type :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('None')
    self.combo1.addItem('Blur')
    self.combo1.addItem('Gaussian')
    self.combo1.addItem('Median')
    self.combo1.addItem('Bilateral')
    self.combo1.currentIndexChanged[int].connect(self.filterChanged)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtWidgets.QLabel('Size :'))   
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(99)
    self.spin1.setValue(1)
    self.spin1.setSingleStep(2)
    self.spin1.setEnabled(False)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('Sigma0 :'))   
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(0)
    self.spin2.setMaximum(300)
    self.spin2.setValue(0)
    self.spin2.setEnabled(False)
    self.spin2.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin2)   
    vbox.addWidget(QtWidgets.QLabel('Sigma1 :'))   
    self.spin3 = QtWidgets.QSpinBox()
    self.spin3.setMinimum(0)
    self.spin3.setMaximum(300)
    self.spin3.setValue(0)
    self.spin3.setEnabled(False)
    self.spin3.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin3)
    vbox.addStretch()

  def setInfo(self, info):
    self.insitu = False
    self.combo1.setCurrentIndex(info['FilterType'])
    self.spin1.setValue(info['FilterSize'])    
    self.spin2.setValue(info['FilterSigma0'])
    self.spin3.setValue(info['FilterSigma1'])
    self.insitu = True
    
  def getInfo(self):
    info = {}
    info['FilterType'] = self.combo1.currentIndex()
    info['FilterSize'] = self.spin1.value()
    info['FilterSigma0'] = self.spin2.value()
    info['FilterSigma1'] = self.spin3.value()
    return info

  def filterChanged(self):
    ftype = self.combo1.currentIndex()
    flag = [[False, False, False], [True, False, False],\
      [True, True, True], [True, False, False], [True, True, True]]
    self.spin1.setEnabled(flag[ftype][0])
    self.spin2.setEnabled(flag[ftype][1])
    self.spin3.setEnabled(flag[ftype][2])
    self.setImage()

  def setImage(self):
    if self.insitu:
      dst_img = self.Process(self.src_img, self.getInfo())
      if dst_img is not None:
        self.imageChanged.emit(dst_img)

  @staticmethod
  def Process(src_img, info):
    if src_img is None:
      return None
    ftype = info['FilterType']
    size = info['FilterSize']
    sigma0 = float(info['FilterSigma0'])
    sigma1 = float(info['FilterSigma1'])
    if ftype == 1:
      return cv2.blur(src_img, (size, size))
    elif ftype == 2:
      return cv2.GaussianBlur(src_img, (size, size), sigma0, sigma1)
    elif ftype == 3:
      return cv2.medianBlur(src_img, size)
    elif ftype == 4:
      return cv2.bilateralFilter(src_img, size, sigma0, sigma1)
    else:
      return src_img.copy()

"""
Threshold Widget
"""  
class ThresholdWidget(QtWidgets.QWidget):
  imageChanged = QtCore.pyqtSignal(np.ndarray)

  def __init__(self, parent):
    super(ThresholdWidget, self).__init__(parent)
    self.src_img = None
    self.dst_img = None
    self.insitu = True
    vbox = QtWidgets.QVBoxLayout(self)
    vbox.addWidget(QtWidgets.QLabel('Type :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('Simple')
    self.combo1.addItem('Otsu')
    self.combo1.addItem('Adaptive Mean')
    self.combo1.addItem('Adaptive Gauss')
    self.combo1.currentIndexChanged[int].connect(self.methodChanged)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtWidgets.QLabel('Threshold :'))   
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(255)
    self.spin1.setValue(127)
    self.spin1.setEnabled(True)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('Adaptive Block Size :'))   
    self.spin2 = QtWidgets.QSpinBox()   
    self.spin2.setMinimum(3)
    self.spin2.setMaximum(999)
    self.spin2.setValue(123)   
    self.spin2.setSingleStep(2)
    self.spin2.setEnabled(False)
    self.spin2.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin2)   
    vbox.addWidget(QtWidgets.QLabel('Adaptive Parm :'))   
    self.spin3 = QtWidgets.QSpinBox()
    self.spin3.setMinimum(-128)
    self.spin3.setMaximum(128)
    self.spin3.setValue(0)
    self.spin3.setEnabled(False)
    self.spin3.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin3)
    self.check1 = QtWidgets.QCheckBox('Invert')
    self.check1.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check1)
    vbox.addStretch()    

  def setInfo(self, info):
    self.insitu = False
    self.combo1.setCurrentIndex(info['ThreshType'])
    self.spin1.setValue(info['ThreshThreshold'])    
    self.spin2.setValue(info['ThreshAdaptiveBlockSize'])
    self.spin3.setValue(info['ThreshAdaptiveParm'])
    self.check1.setChecked(info['ThreshInvert'])
    self.insitu = True
    
  def getInfo(self):
    info = {}
    info['ThreshType'] = self.combo1.currentIndex()
    info['ThreshThreshold'] = self.spin1.value()
    info['ThreshAdaptiveBlockSize'] = self.spin2.value()
    info['ThreshAdaptiveParm'] = self.spin3.value()
    info['ThreshInvert'] = self.check1.isChecked()
    return info

  def methodChanged(self):
    ttype = self.combo1.currentIndex()
    flag = [[True, False, False], [False, False, False],\
      [False, True, True], [False, True, True]]
    self.spin1.setEnabled(flag[ttype][0])
    self.spin2.setEnabled(flag[ttype][1])
    self.spin3.setEnabled(flag[ttype][2])
    self.setImage()

  def setImage(self):
    if self.insitu:
      dst_img = self.Process(self.src_img, self.getInfo())
      if dst_img is not None:
        self.imageChanged.emit(dst_img)

  @staticmethod
  def Process(src_img, info):
    if src_img is None:
      return None
    if len(src_img.shape) == 3:
      src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    ttype = info['ThreshType']
    thres = info['ThreshThreshold']
    bsize = info['ThreshAdaptiveBlockSize']
    parm = info['ThreshAdaptiveParm']
    inv = info['ThreshInvert']
    if inv:
      sty = cv2.THRESH_BINARY_INV
    else:
      sty = cv2.THRESH_BINARY  
    if ttype == 0:
      ret, dst_img = cv2.threshold(src_img, thres, 255, sty)
    elif ttype == 1:
      ret, dst_img = cv2.threshold(src_img, 0, 255, sty + cv2.THRESH_OTSU)
    elif ttype == 2:
      dst_img = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, sty, bsize, parm)
    elif ttype == 3:
      dst_img = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, sty, bsize, parm)
    return dst_img          

"""
Morphology Widget
"""  
class MorphologyWidget(QtWidgets.QWidget):
  imageChanged = QtCore.pyqtSignal(np.ndarray)
  
  def __init__(self, parent):
    super(MorphologyWidget, self).__init__(parent)
    self.src_img = None
    self.insitu = True
    vbox = QtWidgets.QVBoxLayout(self)
    vbox.addWidget(QtWidgets.QLabel('Type :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('Opening')
    self.combo1.addItem('Closing')
    self.combo1.currentIndexChanged[int].connect(self.setImage)
    vbox.addWidget(self.combo1)  
    vbox.addWidget(QtWidgets.QLabel('Iterations :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(32)
    self.spin1.setValue(0)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addStretch() 
    self.label1 = QtWidgets.QLabel('Black :')
    vbox.addWidget(self.label1)
    self.label2 = QtWidgets.QLabel('White :')
    vbox.addWidget(self.label2)

  def setInfo(self, info):
    self.insitu = False
    self.combo1.setCurrentIndex(info['MorpholType'])
    self.spin1.setValue(info['MorpholIterations'])
    self.insitu = True
    
  def getInfo(self):
    info = {}
    info['MorpholType'] = self.combo1.currentIndex()
    info['MorpholIterations'] = self.spin1.value()
    return info

  def setImage(self):
    if self.insitu:
      dst_img = self.Process(self.src_img, self.getInfo())
      if dst_img is not None:
        self.imageChanged.emit(dst_img)
        hist = cv2.calcHist([dst_img], [0], None, [256], [0,256])
        tot = hist[0][0] + hist[255][0]
        self.label1.setText('Black : ' + str(100.0 * hist[0][0] / tot) + ' %')
        self.label2.setText('White : ' + str(100.0 * hist[255][0] / tot) + ' %')

  def clearLabel(self):
    self.label1.setText('Black :')
    self.label2.setText('White :')

  @staticmethod
  def Process(src_img, info):
    if src_img is None:
      return None
    mtype = info['MorpholType']
    it = info['MorpholIterations']
    kernel = np.ones((3,3), np.uint8)
    if it > 0:
      if mtype == 0:
        return cv2.morphologyEx(src_img, cv2.MORPH_OPEN, kernel, iterations=it)
      elif mtype == 1:
        return cv2.morphologyEx(src_img, cv2.MORPH_CLOSE, kernel, iterations=it)
    else:
      return src_img

"""
Modify Widget
"""  
class ModifyWidget(QtWidgets.QWidget):
  imageChanged = QtCore.pyqtSignal(np.ndarray)
  mouseModeChanged = QtCore.pyqtSignal(int)

  def __init__(self, parent):
    super(ModifyWidget, self).__init__(parent)
    self.src_img = None
    self.org_img = None
    self.mod_objects = []
    self.insitu = True
    vbox = QtWidgets.QVBoxLayout(self)
    vbox.addWidget(QtWidgets.QLabel('Source weight :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(255)
    self.spin1.setValue(255)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    self.check1 = QtWidgets.QCheckBox('Contour draw')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check1)
    self.check2 = QtWidgets.QCheckBox('Center draw')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check2)
    self.check4 = QtWidgets.QCheckBox('BoundRect draw')
    self.check4.setChecked(True)
    self.check4.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check4)    
    self.check3 = QtWidgets.QCheckBox('Modify draw')
    self.check3.setChecked(True)
    self.check3.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check3)
    vbox.addWidget(QtWidgets.QLabel('Modify color :'))
    hbox1 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.group1 = QtWidgets.QButtonGroup()
    button1 = QtWidgets.QPushButton('Black')
    button1.setCheckable(True)
    button1.setChecked(True)
    self.group1.addButton(button1, 0)
    hbox1.addWidget(button1)
    button2 = QtWidgets.QPushButton('White')
    button2.setCheckable(True)
    self.group1.addButton(button2, 1)
    hbox1.addWidget(button2)    
    vbox.addWidget(QtWidgets.QLabel('Modify mode :'))
    hbox2 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox2)
    self.group2 = QtWidgets.QButtonGroup()
    self.group2.buttonClicked.connect(self.setMouseMode)
    button3 = QtWidgets.QPushButton('Line')
    button3.setCheckable(True)
    button3.setChecked(True)
    self.group2.addButton(button3, 0)
    hbox2.addWidget(button3)
    button4 = QtWidgets.QPushButton('Rect')
    button4.setCheckable(True)
    self.group2.addButton(button4, 1)
    hbox2.addWidget(button4)
    button5 = QtWidgets.QPushButton('Clear')
    button5.setCheckable(True)
    self.group2.addButton(button5, 2)
    hbox2.addWidget(button5)   
    vbox.addWidget(QtWidgets.QLabel('Line width :'))
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(1)
    self.spin2.setMaximum(16)
    self.spin2.setValue(1)
    vbox.addWidget(self.spin2)
    vbox.addStretch()

  def addLine(self, sx, sy, ex, ey):
    thickness = self.spin2.value()
    if self.group1.checkedId() == 0:
      self.mod_objects.append([0, int(sx), int(sy), int(ex), int(ey), 0, thickness])
    elif self.group1.checkedId() == 1:
      self.mod_objects.append([0, int(sx), int(sy), int(ex), int(ey), 255, thickness])  
    self.setImage()

  def addRect(self, sx, sy, ex, ey):
    if self.group1.checkedId() == 0:
      self.mod_objects.append([1, int(sx), int(sy), int(ex), int(ey), 0, -1])
    elif self.group1.checkedId() == 1:
      self.mod_objects.append([1, int(sx), int(sy), int(ex), int(ey), 255, -1])  
    self.setImage()

  def clearRect(self, sx, sy, ex, ey):
    ids = []
    for i in range(len(self.mod_objects)):
      obj = self.mod_objects[i]
      if obj[1] > sx and obj[2] > sy and obj[3] < ex and obj[4] < ey:
        ids.append(i)
    for i in ids[::-1]:
      self.mod_objects.pop(i)
    self.setImage()

  def setMouseMode(self):
    if self.insitu:
      if self.group2.checkedId() == 0:
        self.mouseModeChanged.emit(ImageScene.mouseLine)
      elif self.group2.checkedId() == 1:
        self.mouseModeChanged.emit(ImageScene.mouseRect)
      elif self.group2.checkedId() == 2:
        self.mouseModeChanged.emit(ImageScene.mouseClear)

  def setInfo(self, info):
    self.insitu = False
    self.spin1.setValue(info['ModSourceWeight'])
    self.check1.setChecked(info['ModContourDraw'])
    self.check2.setChecked(info['ModCenterDraw'])
    self.check3.setChecked(info['ModModifyDraw'])
    self.check4.setChecked(info['ModBoundRectDraw'])
    self.group1.button(info['ModModifyColor']).setChecked(True)
    self.group2.button(info['ModModifyMode']).setChecked(True)
    self.spin2.setValue(info['ModLineWidth'])
    self.mod_objects = info['ModModifyObjects']  
    self.insitu = True
    
  def getInfo(self, with_mod=True):
    info = {}
    info['ModSourceWeight'] = self.spin1.value()
    info['ModContourDraw'] = self.check1.isChecked()
    info['ModCenterDraw'] = self.check2.isChecked()
    info['ModModifyDraw'] = self.check3.isChecked()
    info['ModBoundRectDraw'] = self.check4.isChecked()
    info['ModModifyColor'] = self.group1.checkedId()
    info['ModModifyMode'] = self.group2.checkedId()
    info['ModLineWidth'] = self.spin2.value()
    if with_mod:
      info['ModModifyObjects'] = self.mod_objects
    else:
      info['ModModifyObjects'] = []
    return info

  def setImage(self):
    if self.insitu:
      dst_img = self.Process(self.src_img, self.org_img, self.getInfo())
      if dst_img is not None:
        self.imageChanged.emit(dst_img)

  @staticmethod
  def ProcessMod(src_img, info):
    if src_img is None:
      return None
    mod_img = src_img.copy()
    for obj in info['ModModifyObjects']:
      if obj[0] == 0:
        cv2.line(mod_img, (obj[1], obj[2]), (obj[3], obj[4]), (obj[5], obj[5], obj[5]), thickness=obj[6])
      elif obj[0] == 1:
        cv2.rectangle(mod_img, (obj[1], obj[2]), (obj[3], obj[4]), (obj[5], obj[5], obj[5]), thickness=obj[6])
    return mod_img

  @staticmethod
  def Process(src_img, org_img, info):
    if src_img is None:
      return None
    mod_img = ModifyWidget.ProcessMod(src_img, info)
    conts, hier = cv2.findContours(mod_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_img = np.zeros(org_img.shape, dtype=np.uint8)
    subw = 255 - info['ModSourceWeight']
    cv2.subtract(org_img, (subw, subw, subw, subw), draw_img)
    if info['ModModifyDraw']:
      for obj in info['ModModifyObjects']:
        if obj[0] == 0:
          cv2.line(draw_img, (obj[1], obj[2]), (obj[3], obj[4]), (obj[5], obj[5], obj[5]), thickness=obj[6])
        elif obj[0] == 1:
          cv2.rectangle(draw_img, (obj[1], obj[2]), (obj[3], obj[4]), (obj[5], obj[5], obj[5]), thickness=obj[6])
    if info['ModCenterDraw']:
      for cont in conts:
        mom = cv2.moments(cont)
        if mom['m00'] != 0:
          x = int(mom['m10'] / mom['m00'])
          y = int(mom['m01'] / mom['m00'])
          cv2.line(draw_img, (x, y - 3), (x, y + 3), (0, 0, 255))
          cv2.line(draw_img, (x - 3, y), (x + 3, y), (0, 0, 255))
    if info['ModBoundRectDraw']:
      for cont in conts:
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (255, 0, 0), 1)
    if info['ModContourDraw']:
      cv2.drawContours(draw_img, conts, -1, (255,255,0), 1)
    return draw_img

"""
Misc Functions
"""
def PlotUnitText(unit):
  if unit == 'um':
    return '$\mu$m'
  else:
    return unit

def BinaryImage(info):
  org_img = cv2.imread(info['FileName'], 1)
  roi_img = FileWidget.Process(org_img, info)
  cont_img = ContrastWidget.Process(roi_img, info)
  filter_img = FilterWidget.Process(cont_img, info)
  thresh_img = ThresholdWidget.Process(filter_img, info)
  morphol_img = MorphologyWidget.Process(thresh_img, info)
  mod_img = ModifyWidget.ProcessMod(morphol_img, info)
  return mod_img

"""
Measure IMFP Thread
"""
class MeasureIMFPThread(QtCore.QThread):
  Progress = QtCore.pyqtSignal(int, str, list)  
  
  def __init__(self, parent=None):   
    super(MeasureIMFPThread, self).__init__(parent)
    self.barrier = 255
    self.dpix = None
    self.nsample = 0
    self.seed = 0
    self.freq = None
    self.stat = None
    self.psmax = None
    self.image_info = None

  def setup(self, barrier, nsample, pixmax, seed, image_info):
    self.image_info = []
    self.psmax = [0.0, 'pixel']
    for info in image_info:
      if info['FileMeasure']:
        self.image_info.append(info)
        ps = FileWidget.PixelSize(info)
        if ps[0] > self.psmax[0]:
          self.psmax = ps
    self.barrier = barrier
    self.nsample = nsample
    self.freq = np.zeros((2, pixmax), dtype=np.uint32)
    self.seed = seed

  def run(self):
    inc = 100 / len(self.image_info)
    for info in self.image_info:
      fname = info['FileName']
      ps = FileWidget.PixelSize(info)
      filename = QtCore.QFileInfo(fname).fileName()
      bimg = BinaryImage(info)
      dpix = self.psmax[0] / ps[0]
      MPImfp.measure(bimg, self.barrier, self.freq[0], dpix, self.nsample, self.seed, 0)
      self.seed = MPImfp.measure(bimg, self.barrier, self.freq[1], dpix, self.nsample, self.seed, 1)
      self.Progress.emit(inc, filename, ps)
    self.stat = [self.Statistics(self.freq[0], self.psmax),\
                 self.Statistics(self.freq[1], self.psmax)]
    self.finished.emit()

  @staticmethod
  def RelativeFrequency(freq):
    return np.array(freq, dtype=np.float) / np.sum(freq)

  @staticmethod
  def Statistics(freq, ps):
    tot = np.sum(freq)
    rfreq = np.array(freq, dtype=np.float) / tot
    length = np.arange(len(freq)) * ps[0]
    ave = np.sum(length * rfreq)
    var = np.sum(np.power(length - ave, 2) * rfreq)
    std = m.sqrt(var)
    return [tot, ave, std]

"""
IMFP Dialog
"""
class IMFPDialog(QtWidgets.QDialog):
  def __init__(self, parent):
    QtWidgets.QDialog.__init__(self, parent)
    self.setWindowTitle("IMFP")
    self.parent = parent
    self.insitu = True
    self.freq = None
    self.stat = None
    self.psmax = None
    self.measure = MeasureIMFPThread()
    self.measure.finished.connect(self.measureFinish)
    self.measure.Progress.connect(self.measureProgress)
#    self.pdlg = QtWidgets.QProgressDialog(self)
#    self.pdlg.setWindowTitle("Measuring IMFP ...")
#    self.pdlg.canceled.connect(self.measureCancel)
    hbox = QtWidgets.QHBoxLayout(self)
    vbox = QtWidgets.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = QtWidgets.QGraphicsView()
    self.scene = QtWidgets.QGraphicsScene()
    self.viewer.setScene(self.scene)
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.scene.addWidget(self.canvas)
    hbox.addWidget(self.viewer)
    vbox.addWidget(QtWidgets.QLabel('Barrier :'))
    self.combo0 = QtWidgets.QComboBox()
    self.combo0.addItem('White')  
    self.combo0.addItem('Black')
    vbox.addWidget(self.combo0)
    vbox.addWidget(QtWidgets.QLabel('NSample (x10000) :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(1000)
    self.spin1.setValue(100)      
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('NClass :'))
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(100)
    self.spin2.setMaximum(10000)
    self.spin2.setValue(5000)      
    vbox.addWidget(self.spin2)
    vbox.addWidget(QtWidgets.QLabel('Seed :'))
    self.line1 = QtWidgets.QLineEdit()
    seed = random.randint(1, 1000000000)
    self.line1.setText(str(seed))
    vbox.addWidget(self.line1)
    self.button1 = QtWidgets.QPushButton('Measure')
    self.button1.clicked[bool].connect(self.measureIMFP)
    vbox.addWidget(self.button1)
    self.treeview = QtWidgets.QTreeView()
    self.treemodel = QtGui.QStandardItemModel()
    self.treemodel.setHorizontalHeaderLabels(['Files', 'PS', 'Unit'])
    self.treeview.setModel(self.treemodel)
    self.treeview.header().setStretchLastSection(False)
#    self.treeview.header().setResizeMode(0, QtWidgets.QHeaderView.Stretch)
#    self.treeview.header().setResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
#    self.treeview.header().setResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
    vbox.addWidget(self.treeview)
    vbox.addWidget(QtWidgets.QLabel('Type :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('Single')  
    self.combo1.addItem('Double')
    self.combo1.currentIndexChanged.connect(self.drawGraph)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtWidgets.QLabel('Plot NClass :'))
    self.spin3 = QtWidgets.QSpinBox()
    self.spin3.setMinimum(10)
    self.spin3.setMaximum(10000)
    self.spin3.setValue(5000)
    self.spin3.valueChanged.connect(self.drawGraph)     
    vbox.addWidget(self.spin3)
    self.check1 = QtWidgets.QCheckBox('Relative Frequency')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check1)
    self.check2 = QtWidgets.QCheckBox('Show Statistics')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check2)
    vbox.addWidget(QtWidgets.QLabel('DPI :'))
    self.spin4 = QtWidgets.QSpinBox()
    self.spin4.setMinimum(10)
    self.spin4.setMaximum(3000)
    self.spin4.setValue(100)
    vbox.addWidget(self.spin4)
    vbox.addStretch()
    hbox1 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button2 = QtWidgets.QPushButton('Save CSV')
    self.button2.clicked[bool].connect(self.saveCSV)
    hbox1.addWidget(self.button2)
    self.button3 = QtWidgets.QPushButton('Save Graph')
    self.button3.clicked[bool].connect(self.saveGraph)
    hbox1.addWidget(self.button3)
    self.button4 = QtWidgets.QPushButton('Close')
    self.button4.clicked[bool].connect(self.close)
    hbox1.addWidget(self.button4)

  def setInfo(self, info):
    self.combo0.setCurrentIndex(info['Barrier'])
    self.spin1.setValue(info['NSample'])    
    self.spin2.setValue(info['NClass'])
    self.line1.setText(info['Seed'])
    self.insitu = False
    self.combo1.setCurrentIndex(info['Type'])
    self.spin3.setValue(info['PlotClassMax'])
    self.check1.setChecked(info['RelativeFrequency'])
    self.check2.setChecked(info['ShowStatistics'])
    self.insitu = True
    self.spin4.setValue(info['DPI'])
    self.treemodel.removeRows(0, self.treemodel.rowCount())

  def getInfo(self):
    info = {}
    info['Barrier'] = self.combo0.currentIndex()
    info['NSample'] = self.spin1.value()
    info['NClass'] = self.spin2.value()
    info['Seed'] = str(self.line1.text())
    info['Type'] = self.combo1.currentIndex()
    info['PlotClassMax'] = self.spin3.value()
    info['RelativeFrequency'] = self.check1.isChecked()
    info['ShowStatistics'] = self.check2.isChecked()
    info['DPI'] = self.spin4.value()
    return info

  def measureIMFP(self):
    if len(self.parent.image_info) > 0:
      if self.combo0.currentIndex() == 1:
        barrier = 0
      else:
        barrier = 255
      nsample = self.spin1.value() * 10000
      pixmax = self.spin2.value()
      seed = int(self.line1.text())
      self.clearFreq()
      self.pdlg = QtWidgets.QProgressDialog(self)
      self.pdlg.setWindowTitle("Measuring IMFP ...")
      self.pdlg.canceled.connect(self.measureCancel)
      self.pdlg.setValue(0)
      self.measure.setup(barrier, nsample, pixmax, seed, self.parent.image_info)        
      self.measure.start()

  def measureProgress(self, inc, filename, ps):
    val = self.pdlg.value()
    self.pdlg.setValue(val + inc)
    root = self.treemodel.invisibleRootItem()
    item1 = QtGui.QStandardItem(filename)
    item1.setEditable(False)
    item2 = QtGui.QStandardItem('%.3f' % ps[0])
    item2.setEditable(False)
    item3 = QtGui.QStandardItem(ps[1])
    item3.setEditable(False)
    root.appendRow([item1, item2, item3])

  def measureFinish(self):
    self.pdlg.close()
    if self.measure.freq is not None and self.measure.stat is not None:
      self.freq = self.measure.freq
      self.stat = self.measure.stat
      self.psmax = self.measure.psmax
      self.drawGraph()

  def measureCancel(self):
    self.measure.terminate()

  def drawIMFP(self, freq, pixmax, dsize, xlabel, ylabel, stat):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    length = np.arange(freq.size) * dsize
    ax.plot(length, freq, 'k-', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, pixmax * dsize)
    if stat is not None:
      ax.text(0.6, 0.95, 'NSample : ' + str(stat[0]), transform=ax.transAxes)
      ax.text(0.6, 0.9, 'Mean : ' + str(stat[1]), transform=ax.transAxes) 
      ax.text(0.6, 0.85, 'STD : ' + str(stat[2]), transform=ax.transAxes)
    self.canvas.draw()

  def drawGraph(self):
    if self.freq is None or self.insitu == False:
      return
    if self.combo1.currentIndex() == 0:
      find = 0
      xlabel = 'Single length (%s)' % PlotUnitText(self.psmax[1])
    elif self.combo1.currentIndex() == 1:
      find = 1
      xlabel = 'Double length (%s)' % PlotUnitText(self.psmax[1])
    if self.check1.isChecked():
      freq = MeasureIMFPThread.RelativeFrequency(self.freq[find])
      ylabel = 'Relative Frequency'
    else:
      freq = self.freq[find]
      ylabel = 'Frequency'
    if self.check2.isChecked():
      stat = self.stat[find]
    else:
      stat = None
    pixmax = self.spin3.value()
    self.drawIMFP(freq, pixmax, self.psmax[0], xlabel, ylabel, stat)

  def saveCSV(self):
    if self.freq is None:
      return
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV Files (*.csv);;All Files (*.*)')[0]
    if fname:
      fout = open(fname, 'w')
      nimg = self.treemodel.rowCount()
      fout.write('Images,' + str(nimg) + '\n')      
      fout.write('ImageID, FileName, PixelSize, Unit\n')
      for i in range(nimg):
        fname = self.treemodel.item(i,0).text()
        ps = self.treemodel.item(i,1).text()
        unit = self.treemodel.item(i,2).text()
        fout.write('%d, %s, %s, %s\n' % (i, fname, ps, unit))
      fout.write('Statistics, Total, Mean, STD\n')
      tt = ['Single', 'Double']
      c = 0 
      for st in self.stat:
        fout.write('%s, %d, %f, %f\n' % (tt[c], st[0], st[1], st[2]))
        c += 1
      fout.write('Class, Length, SingleF, SingleRF, DoubleF, DoubleRF\n')
      f0 = self.freq[0]
      f1 = self.freq[1]
      rf0 = MeasureIMFPThread.RelativeFrequency(f0)
      rf1 = MeasureIMFPThread.RelativeFrequency(f1)
      for i in range(len(f0)):
        length = i * self.psmax[0]
        fout.write('%d, %f, %d, %f, %d, %f\n' % (i, length, f0[i], rf0[i], f1[i], rf1[i]))

  def saveGraph(self):
    if self.freq is None:
      return
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Graph', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')[0]
    if fname:
      self.figure.savefig(str(fname), dpi=self.spin4.value())    

  def clearFreq(self):
    self.freq = None
    self.stat = None
    self.psmax = None
    self.treemodel.removeRows(0, self.treemodel.rowCount())
    self.figure.clf()
    self.canvas.draw()

"""
Measure LN2D Thread
"""
class MeasureLN2DThread(QtCore.QThread):
  Progress = QtCore.pyqtSignal(int, str, list)  
  
  def __init__(self, parent=None):   
    super(MeasureLN2DThread, self).__init__(parent)
    self.nsample = 0
    self.freq = None
    self.image_info = None
    self.ln2d = None
    self.af = 0
    self.stat = None

  def setup(self, nsample, lnmax, seed, image_info):
    self.image_info = []
    for info in image_info:
      if info['FileMeasure']:
        self.image_info.append(info)
    self.nsample = nsample
    self.freq = np.zeros((2, lnmax), dtype=np.uint32)
    self.ln2d = MPLn23d.ln2d_new(len(self.image_info))
    self.ln2d.seed = seed

  def run(self):
    inc = 100 / len(self.image_info)
    for info in self.image_info:
      fname = info['FileName']
      ps = FileWidget.PixelSize(info)
      filename = QtCore.QFileInfo(fname).fileName()
      img = BinaryImage(info)
      conts, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      self.AddSec(self.ln2d, img, conts, ps)
      self.Progress.emit(inc, filename, ps)
    self.ln2d.measure_gc(self.freq[0])
    self.ln2d.measure_random(self.freq[1], self.nsample)
    self.af = self.ln2d.area_fraction()
    self.stat = [self.Statistics(self.freq[0]), self.Statistics(self.freq[1])]
    self.finished.emit()

  @staticmethod
  def AddSec(ln2d, img, conts, ps):
    sx = img.shape[1] * ps[0]
    sy = img.shape[0] * ps[0]
    sid = ln2d.add_sec(len(conts), sx, sy)
    for cont in conts:
      mom = cv2.moments(cont)
      if mom['m00'] != 0:
        x = mom['m10'] / mom['m00'] * ps[0]
        y = mom['m01'] / mom['m00'] * ps[0]
        area = cv2.contourArea(cont)
        r = m.sqrt(area / m.pi) * ps[0]
        ln2d.add_gc(sid, x, y, r)
    return sid

  @staticmethod
  def RelativeFrequency(freq):
    return np.array(freq, dtype=np.float) / np.sum(freq)

  @staticmethod
  def Statistics(freq):
    tot = np.sum(freq)
    rfreq = np.array(freq, dtype=np.float) / tot
    length = np.arange(len(freq))
    ave = np.sum(length * rfreq)
    var = np.sum(np.power(length - ave, 2) * rfreq)
    return [tot, ave, var]

"""
LN2D Dialog
"""
class LN2DDialog(QtWidgets.QDialog):
  def __init__(self, parent):
    QtWidgets.QDialog.__init__(self, parent)
    self.setWindowTitle("LN2D")
    self.parent = parent
    self.insitu = True
    self.freq = None
    self.stat = None
    self.measure = MeasureLN2DThread()
    self.measure.finished.connect(self.measureFinish)
    self.measure.Progress.connect(self.measureProgress)
    hbox = QtWidgets.QHBoxLayout(self)
    vbox = QtWidgets.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = QtWidgets.QGraphicsView()
    self.scene = QtWidgets.QGraphicsScene()
    self.viewer.setScene(self.scene)
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.scene.addWidget(self.canvas)
    hbox.addWidget(self.viewer)
    vbox.addWidget(QtWidgets.QLabel('NSample (x10000) :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(1000)
    self.spin1.setValue(100)      
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('LN Max :'))
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(10)
    self.spin2.setMaximum(1000)
    self.spin2.setValue(100)
    vbox.addWidget(self.spin2)
    vbox.addWidget(QtWidgets.QLabel('Seed :'))
    self.line1 = QtWidgets.QLineEdit()
    seed = random.randint(1, 1000000000)
    self.line1.setText(str(seed))     
    vbox.addWidget(self.line1)
    self.button1 = QtWidgets.QPushButton('Measure')
    self.button1.clicked[bool].connect(self.measureLN2D)
    vbox.addWidget(self.button1)
    self.treeview = QtWidgets.QTreeView()
    self.treemodel = QtGui.QStandardItemModel()
    self.treemodel.setHorizontalHeaderLabels(['Files', 'PS', 'Unit'])
    self.treeview.setModel(self.treemodel)
    self.treeview.header().setStretchLastSection(False)
#    self.treeview.header().setResizeMode(0, QtWidgets.QHeaderView.Stretch)
#    self.treeview.header().setResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
#    self.treeview.header().setResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
    vbox.addWidget(self.treeview)
    vbox.addWidget(QtWidgets.QLabel('Type :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('Gravity center')  
    self.combo1.addItem('Random')
    self.combo1.currentIndexChanged.connect(self.drawGraph)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtWidgets.QLabel('Plot LN Max :'))
    self.spin3 = QtWidgets.QSpinBox()
    self.spin3.setMinimum(1)
    self.spin3.setMaximum(1000)
    self.spin3.setValue(30)
    self.spin3.valueChanged.connect(self.drawGraph)     
    vbox.addWidget(self.spin3)
    vbox.addWidget(QtWidgets.QLabel('Area fraction :'))
    self.spin4 = QtWidgets.QSpinBox()
    self.spin4.setMinimum(0)
    self.spin4.setMaximum(100)
    self.spin4.setValue(0)
    self.spin4.valueChanged.connect(self.drawGraph)         
    vbox.addWidget(self.spin4)
    self.check1 = QtWidgets.QCheckBox('Relative Frequency')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check1)
    self.check2 = QtWidgets.QCheckBox('Show Statistics')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check2)
    self.check3 = QtWidgets.QCheckBox('Show Reference')
    self.check3.setChecked(True)
    self.check3.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check3)
    vbox.addWidget(QtWidgets.QLabel('DPI :'))
    self.spin5 = QtWidgets.QSpinBox()
    self.spin5.setMinimum(10)
    self.spin5.setMaximum(3000)
    self.spin5.setValue(100)
    vbox.addWidget(self.spin5)    
    vbox.addStretch()
    hbox1 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button2 = QtWidgets.QPushButton('Save CSV')
    self.button2.clicked[bool].connect(self.saveCSV)
    hbox1.addWidget(self.button2)
    self.button3 = QtWidgets.QPushButton('Save Graph')
    self.button3.clicked[bool].connect(self.saveGraph)
    hbox1.addWidget(self.button3)
    self.button4 = QtWidgets.QPushButton('Close')
    self.button4.clicked[bool].connect(self.close)
    hbox1.addWidget(self.button4)

  def setInfo(self, info):
    self.spin1.setValue(info['NSample'])    
    self.spin2.setValue(info['LNMax'])
    self.line1.setText(info['Seed'])
    self.insitu = False
    self.combo1.setCurrentIndex(info['Type'])
    self.spin3.setValue(info['PlotLNMax'])
    self.spin4.setValue(info['AreaFraction'])
    self.check1.setChecked(info['RelativeFrequency'])
    self.check2.setChecked(info['ShowStatistics'])
    self.check3.setChecked(info['ShowReference'])
    self.insitu = True
    self.spin5.setValue(info['DPI'])
    self.treemodel.removeRows(0, self.treemodel.rowCount())

  def getInfo(self):
    info = {}
    info['NSample'] = self.spin1.value()
    info['LNMax'] = self.spin2.value()
    info['Seed'] = str(self.line1.text())
    info['Type'] = self.combo1.currentIndex()
    info['PlotLNMax'] = self.spin3.value()
    info['AreaFraction'] = self.spin4.value()
    info['RelativeFrequency'] = self.check1.isChecked()
    info['ShowStatistics'] = self.check2.isChecked()
    info['ShowReference'] = self.check3.isChecked()
    info['DPI'] = self.spin5.value()
    return info

  def measureLN2D(self):
    if len(self.parent.image_info) > 0:
      nsample = self.spin1.value() * 10000
      lnmax = self.spin2.value()
      seed = int(self.line1.text())
      self.clearFreq()
      self.pdlg = QtWidgets.QProgressDialog(self)
      self.pdlg.setWindowTitle("Measuring LN2D ...")
      self.pdlg.canceled.connect(self.measureCancel)
      self.pdlg.setValue(0)
      self.measure.setup(nsample, lnmax, seed, self.parent.image_info)        
      self.measure.start()

  def measureProgress(self, inc, filename, ps):
    val = self.pdlg.value()
    self.pdlg.setValue(val + inc)
    root = self.treemodel.invisibleRootItem()
    item1 = QtGui.QStandardItem(filename)
    item1.setEditable(False)
    item2 = QtGui.QStandardItem('%.3f' % ps[0])
    item2.setEditable(False)
    item3 = QtGui.QStandardItem(ps[1])
    item3.setEditable(False)
    root.appendRow([item1, item2, item3])

  def measureFinish(self):
    self.pdlg.close()
    if self.measure.freq is not None and self.measure.stat is not None:
      self.freq = self.measure.freq
      self.stat = self.measure.stat
      self.spin4.setValue(int(self.measure.af * 100.0))
      self.drawGraph()

  def measureCancel(self):
    self.measure.terminate()

  def drawLN2D(self, freq, lnmax, xlabel, ylabel, stat, prob):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    ax.bar(np.arange(freq.size) - 0.4, freq, color='white', edgecolor='black')
    if prob != None:
      ax.plot(prob[0], prob[1], 'k-', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, lnmax)
    if stat is not None:
      ax.text(0.5, 0.95, 'NSample : ' + str(stat[0]), transform=ax.transAxes)
      ax.text(0.5, 0.9, 'Av. : ' + str(stat[1]), transform=ax.transAxes)
      ax.text(0.5, 0.85, 'Var. : ' + str(stat[2]), transform=ax.transAxes)
      if prob != None:
        ax.text(0.5, 0.80, 'Ref. Av. : ' + str(prob[2]), transform=ax.transAxes)
        ax.text(0.5, 0.75, 'Ref. Var. : ' + str(prob[3]), transform=ax.transAxes)      
    self.canvas.draw()

  def drawGraph(self):
    if self.freq is None or self.insitu == False:
      return
    if self.combo1.currentIndex() == 0:
      find = 0
      xlabel = 'LN2D'
      prob = self.ln2d_prob(0.01 * self.spin4.value())
    elif self.combo1.currentIndex() == 1:
      find = 1
      xlabel = 'LN2DR'
      prob = self.ln2dr_prob(0.01 * self.spin4.value())
    if self.check1.isChecked():
      freq = self.measure.RelativeFrequency(self.freq[find])
      ylabel = 'Relative Frequency'
    else:
      freq = self.freq[find]
      prob[1] = prob[1] * np.sum(freq)
      ylabel = 'Frequency'
    if self.check2.isChecked():
      stat = self.stat[find]
    else:
      stat = None
    if self.check3.isChecked() is not True:
      prob = None
    lnmax = self.spin3.value()
    self.drawLN2D(freq, lnmax, xlabel, ylabel, stat, prob)

  def poisson_prob(self, a, b):
    x = np.arange(b - 1 + 0.1, 100, 0.1)
    y = np.zeros(x.size, dtype=np.float)
    for i in range(x.size):
      y[i] = m.pow(a, x[i] - b) / m.gamma(x[i] - b + 1) * m.exp(-a)
    return [x, y, a + b, a]

  def ln2d_prob(self, af):
    p, q, r, s = 6.1919, 5.8194, 5.1655, 5.7928
    a = p * (m.exp(-q * af) - 1) + 7
    b = r * (1 - m.exp(-s * af)) + 1
    return self.poisson_prob(a, b)

  def ln2dr_prob(self, af):
    p, q = 5.8277, 6.0755
    a = p * (m.exp(-q * af) - 1) + 7
    b = p * (1 - m.exp(-q * af))    
    return self.poisson_prob(a, b)

  def saveCSV(self):    
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV Files (*.csv);;All Files (*.*)')[0]
    if fname:
      fout = open(fname, 'w')
      nimg = self.treemodel.rowCount()
      fout.write('Images,' + str(nimg) + '\n')      
      fout.write('ImageID, FileName, PixelSize, Unit\n')
      for i in range(nimg):
        fname = self.treemodel.item(i,0).text()
        ps = self.treemodel.item(i,1).text()
        unit = self.treemodel.item(i,2).text()
        fout.write('%d, %s, %s, %s\n' % (i, fname, ps, unit))

      fout.write('Statistics, Total, Average, Variance\n')
      tt = ['GC', 'Random']
      c = 0 
      for st in self.stat:
        fout.write('%s, %d, %f, %f\n' % (tt[c], st[0], st[1], st[2]))
        c += 1
      fout.write('LN, GC, GCRF, Random, RandomRF\n')
      rf0 = self.measure.RelativeFrequency(self.freq[0])
      rf1 = self.measure.RelativeFrequency(self.freq[1])
      for i in range(len(self.freq[0])):
        fout.write('%d, %d, %f, %d, %f\n' % (i, self.freq[0,i], rf0[i], self.freq[1,i], rf1[i]))

  def saveGraph(self):
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Graph', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')[0]
    if fname:
      self.figure.savefig(str(fname), dpi=self.spin5.value())    

  def clearFreq(self):
    self.freq = None
    self.stat = None
    self.treemodel.removeRows(0, self.treemodel.rowCount())
    self.figure.clf()
    self.canvas.draw()

"""
Measure Size Thread
"""
class MeasureSizeThread(QtCore.QThread):
  Progress = QtCore.pyqtSignal(int, str, list)  
  
  def __init__(self, parent=None):   
    super(MeasureSizeThread, self).__init__(parent)
    self.image_info = None
    self.data = None
    self.stat = None
    self.psmin = None
    self.afnd = None

  def setup(self, image_info):
    self.image_info = []
    for info in image_info:
      if info['FileMeasure']:
        self.image_info.append(info)

  def run(self):
    inc = 100 / len(self.image_info)
    self.data = []
    imgarea = 0.0
    self.psmin = [1.0e12, 'pixel']
    ind = 0
    for info in self.image_info:
      fname = info['FileName']
      filename = QtCore.QFileInfo(fname).fileName()
      ps = FileWidget.PixelSize(info)
      img = BinaryImage(info)
      conts, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      self.data.extend(self.Measure(conts, ps, ind))
      if ps[0] < self.psmin[0]:
        self.psmin = ps
      imgarea += img.shape[1] * img.shape[0] * ps[0] * ps[0]
      self.Progress.emit(inc, filename, ps)
      ind += 1
    self.stat = self.Statistics(self.data)
    self.afnd = [self.AreaFraction(self.data, imgarea),\
                 self.NumberDensity(self.data, imgarea)]
    self.finished.emit()

  @staticmethod
  def AreaFraction(data, imgarea):
    ta = 0.0
    for dt in data:
      r = dt[3] / 2.0
      ta += m.pi * r * r
    return 100.0 * ta / imgarea

  @staticmethod
  def NumberDensity(data, imgarea):
    return len(data) / imgarea

  @staticmethod
  def Measure(conts, ps, index):
    data = []
    for cont in conts:
      mom = cv2.moments(cont)
      if mom['m00'] != 0:
        x = mom['m10'] / mom['m00'] * ps[0]
        y = mom['m01'] / mom['m00'] * ps[0]
        area = cv2.contourArea(cont)
        dia = 2.0 * m.sqrt(area / m.pi) * ps[0]
        pos, size, ang = cv2.minAreaRect(cont)
        per = cv2.arcLength(cont, True)
        cir = 4.0 * m.pi * area / (per * per)
        if size[0] >= size[1]:
          ls = size[0] * ps[0]
          ns = size[1] * ps[0]
          asp = size[1] / size[0]
          ang += 90
        else:
          ls = size[1] * ps[0]
          ns = size[0] * ps[0]
          asp = size[0] / size[1]
          ang += 180
        data.append([index, x, y, dia, ls, ns, asp, cir, ang])
    return data

  @staticmethod
  def Statistics(data):
    dia = np.empty(len(data))
    ls = np.empty(len(data))
    ns = np.empty(len(data))
    asp = np.empty(len(data))
    cir = np.empty(len(data))
    ang = np.empty(len(data))
    for i in range(len(data)):
      dia[i] = data[i][3]
      ls[i] = data[i][4]
      ns[i] = data[i][5]      
      asp[i] = data[i][6]
      cir[i] = data[i][7]
      ang[i] = data[i][8]
    stat = []
    stat.append([np.mean(dia), np.std(dia)])
    stat.append([np.mean(ls), np.std(ls)])
    stat.append([np.mean(ns), np.std(ns)])
    stat.append([np.mean(asp), np.std(asp)])
    stat.append([np.mean(cir), np.std(cir)])
    stat.append([np.mean(ang), np.std(ang)])
    return stat
    
  @staticmethod
  def Frequency(data, col, nclass, dsize):
    freq = np.zeros(nclass, dtype=np.uint32)
    for dat in data:
      ind = int(dat[col] / dsize)
      if ind < nclass:
        freq[ind] += 1
    return freq

  @staticmethod
  def RelativeFrequency(data, col, nclass, dsize):
    freq = MeasureSizeThread.Frequency(data, col, nclass, dsize)
    rfreq = np.array(freq, dtype=np.float) / np.sum(freq)
    return rfreq

"""
Size Dialog
"""
class SizeDialog(QtWidgets.QDialog):
  def __init__(self, parent):
    QtWidgets.QDialog.__init__(self, parent)
    self.setWindowTitle("Size")
    self.parent = parent
    self.insitu = True
    self.data = None
    self.stat = None
    self.psmin = None
    self.afnd = None
    self.measure = MeasureSizeThread()
    self.measure.finished.connect(self.measureFinish)
    self.measure.Progress.connect(self.measureProgress)
    hbox = QtWidgets.QHBoxLayout(self)
    vbox = QtWidgets.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = QtWidgets.QGraphicsView()
    self.scene = QtWidgets.QGraphicsScene()
    self.viewer.setScene(self.scene)
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.scene.addWidget(self.canvas)
    hbox.addWidget(self.viewer)
    self.button1 = QtWidgets.QPushButton('Measure')
    self.button1.clicked[bool].connect(self.measureSize)
    vbox.addWidget(self.button1)
    self.treeview = QtWidgets.QTreeView()
    self.treemodel = QtGui.QStandardItemModel()
    self.treemodel.setHorizontalHeaderLabels(['Files', 'PS', 'Unit'])
    self.treeview.setModel(self.treemodel)
    self.treeview.header().setStretchLastSection(False)
#    self.treeview.header().setResizeMode(0, QtWidgets.QHeaderView.Stretch)
#    self.treeview.header().setResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
#    self.treeview.header().setResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
    vbox.addWidget(self.treeview)
    self.labelaf = QtWidgets.QLabel('AF :')
    vbox.addWidget(self.labelaf)
    self.labelnd = QtWidgets.QLabel('ND :')
    vbox.addWidget(self.labelnd)
    vbox.addWidget(QtWidgets.QLabel('Type :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('Diameter')
    self.combo1.addItem('Long side')    
    self.combo1.addItem('Narrow side')
    self.combo1.addItem('Aspect ratio')
    self.combo1.addItem('Circularity')
    self.combo1.addItem('Angle')
    self.combo1.currentIndexChanged.connect(self.drawGraph)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtWidgets.QLabel('NClass :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(500)
    self.spin1.setValue(100)
    self.spin1.valueChanged.connect(self.drawGraph)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('DSize :'))
    self.line1 = QtWidgets.QLineEdit()
    self.line1.setValidator(QtGui.QDoubleValidator())
    self.line1.textChanged[str].connect(self.drawGraph)
    self.line1.setText('1.0')
    vbox.addWidget(self.line1)
    self.check1 = QtWidgets.QCheckBox('Relative Frequency')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check1)
    self.check2 = QtWidgets.QCheckBox('Show Statistics')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check2)
    vbox.addWidget(QtWidgets.QLabel('DPI :'))
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(10)
    self.spin2.setMaximum(3000)
    self.spin2.setValue(100)
    vbox.addWidget(self.spin2)
    vbox.addStretch()    
    hbox1 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button2 = QtWidgets.QPushButton('Save CSV')
    self.button2.clicked[bool].connect(self.saveCSV)
    hbox1.addWidget(self.button2)
    self.button3 = QtWidgets.QPushButton('Save Graph')
    self.button3.clicked[bool].connect(self.saveGraph)
    hbox1.addWidget(self.button3)
    self.button4 = QtWidgets.QPushButton('Close')
    self.button4.clicked[bool].connect(self.close)
    hbox1.addWidget(self.button4)

  def setInfo(self, info):
    self.insitu = False
    self.combo1.setCurrentIndex(info['Type'])
    self.spin1.setValue(info['NClass'])
    self.line1.setText(info['DSize'])
    self.check1.setChecked(info['RelativeFrequency'])
    self.check2.setChecked(info['ShowStatistics'])
    self.insitu = True
    self.spin2.setValue(info['DPI'])
    self.treemodel.removeRows(0, self.treemodel.rowCount())

  def getInfo(self):
    info = {}
    info['Type'] = self.combo1.currentIndex()
    info['NClass'] = self.spin1.value()
    info['DSize'] = str(self.line1.text())
    info['RelativeFrequency'] = self.check1.isChecked()
    info['ShowStatistics'] = self.check2.isChecked()
    info['DPI'] = self.spin2.value()
    return info

  def measureSize(self):
    if len(self.parent.image_info) > 0:
      self.clearData()
      self.pdlg = QtWidgets.QProgressDialog(self)
      self.pdlg.setWindowTitle("Measuring Size ...")
      self.pdlg.canceled.connect(self.measureCancel)
      self.pdlg.setValue(0)
      self.measure.setup(self.parent.image_info)
      self.measure.start()

  def measureProgress(self, inc, filename, ps):
    val = self.pdlg.value()
    self.pdlg.setValue(val + inc)
    root = self.treemodel.invisibleRootItem()
    item1 = QtGui.QStandardItem(filename)
    item1.setEditable(False)
    item2 = QtGui.QStandardItem('%.3f' % ps[0])
    item2.setEditable(False)
    item3 = QtGui.QStandardItem(ps[1])
    item3.setEditable(False)
    root.appendRow([item1, item2, item3])

  def measureFinish(self):
    self.pdlg.close()
    if self.measure.data is not None and self.measure.stat is not None:
      self.data = self.measure.data
      self.stat = self.measure.stat
      self.psmin = self.measure.psmin
      self.afnd = self.measure.afnd
      self.labelaf.setText('AF : %f' % self.afnd[0])
      self.labelnd.setText('ND : %f' % self.afnd[1])
      self.line1.setText('%.3f' % self.psmin[0])
      self.drawGraph()

  def measureCancel(self):
    self.measure.terminate()

  def drawSize(self, freq, nclass, dsize, xlabel, ylabel, nsample, stat):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    length = np.arange(freq.size) * dsize
    ax.bar(length, freq, width=dsize, color='white', edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, nclass * dsize)
    if stat is not None:
      ax.text(0.5, 0.95, 'NSample : ' + str(nsample), transform=ax.transAxes)
      ax.text(0.5, 0.9, 'Mean : ' + str(stat[0]), transform=ax.transAxes) 
      ax.text(0.5, 0.85, 'STD : ' + str(stat[1]), transform=ax.transAxes)
    self.canvas.draw()

  def drawGraph(self):
    if self.data is None or self.insitu == False:
      return
    if len(self.line1.text()) == 0 or float(self.line1.text()) < 1.0e-3:
      return
    if self.spin1.value() < 2:
      return
    if self.check2.isChecked():
      stat = self.stat
    else:
      stat = [None, None, None, None, None]
    if self.combo1.currentIndex() == 0:
      row = 3
      nclass = self.spin1.value()
      dsize = float(self.line1.text())
      xlabel = 'Diameter (%s)' % PlotUnitText(self.psmin[1])
    elif self.combo1.currentIndex() == 1:
      row = 4
      nclass = self.spin1.value()
      dsize = float(self.line1.text())
      xlabel = 'Size of long side (%s)' % PlotUnitText(self.psmin[1])
    elif self.combo1.currentIndex() == 2:
      row = 5
      nclass = self.spin1.value()
      dsize = float(self.line1.text())
      xlabel = 'Size of narrow side (%s)' % PlotUnitText(self.psmin[1])
    elif self.combo1.currentIndex() == 3:
      row = 6
      nclass = 21
      dsize = 1.0 / (nclass - 1)
      xlabel = 'Aspect ratio'
    elif self.combo1.currentIndex() == 4:
      row = 7
      nclass = 21
      dsize = 1.0 / (nclass - 1)
      xlabel = 'Circularity'
    elif self.combo1.currentIndex() == 5:
      row = 8
      nclass = 19
      dsize = 180.0 / (nclass - 1)
      xlabel = 'Angle (degree)'
    nsample = len(self.data)
    if self.check1.isChecked():
      freq = MeasureSizeThread.RelativeFrequency(self.data, row, nclass, dsize)
      self.drawSize(freq, nclass, dsize, xlabel, 'Relative Frequency', nsample, stat[row - 3])  
    else:
      freq = MeasureSizeThread.Frequency(self.data, row, nclass, dsize)
      self.drawSize(freq, nclass, dsize, xlabel, 'Frequency', nsample, stat[row - 3])

  def saveCSV(self):    
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV Files (*.csv);;All Files (*.*)')[0]
    if fname:
      fout = open(fname, 'w')
      nimg = self.treemodel.rowCount()
      fout.write('Images,' + str(nimg) + '\n')
      fout.write('ImageID, FileName, PixelSize, Unit\n')
      for i in range(nimg):
        fname = self.treemodel.item(i,0).text()
        ps = self.treemodel.item(i,1).text()
        unit = self.treemodel.item(i,2).text()
        fout.write('%d, %s, %s, %s\n' % (i, fname, ps, unit))
      fout.write('AreaFraction, %f\n' % self.afnd[0])
      fout.write('NumberDensity, %f\n' % self.afnd[1])
      fout.write('Statistics, NSample, Mean, STD\n')
      tt = ['Diameter', 'LongSide', 'NarrowSide', 'AspectRatio', 'Circularity', 'Angle']
      c = 0 
      for st in self.stat:
        fout.write('%s, %d, %f, %f\n' % (tt[c], len(self.data), st[0], st[1]))
        c += 1
      fout.write('ImageID, X, Y, Diameter, LongSide, NarrowSide, AspectRatio, Circularity, Angle\n')
      for dat in self.data:
        fout.write('%d, %f, %f, %f, %f, %f, %f, %f, %f\n' % \
                   (dat[0], dat[1], dat[2], dat[3], dat[4], dat[5], dat[6], dat[7], dat[8]))

  def saveGraph(self):
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Graph', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')[0]
    if fname:
      self.figure.savefig(str(fname), dpi=self.spin2.value())
      
  def clearData(self):
    self.data = None
    self.stat = None
    self.treemodel.removeRows(0, self.treemodel.rowCount())
    self.figure.clf()
    self.canvas.draw()

"""
Measure Each Thread
"""
class MeasureEachThread(QtCore.QThread):
  Progress = QtCore.pyqtSignal(int)

  def __init__(self, parent=None):
    super(MeasureEachThread, self).__init__(parent)
    self.clf = None
    self.psmax = None
    self.image_info = None
    self.measure_info = None
    self.features = None    

  def setup(self, image_info):
    self.image_info = image_info
    #self.psmax = psmax
    #self.measure_info = measure_info

  def run(self):
    tot = len(self.image_info)
    inc = 100/tot
    self.features = []
    for info in self.image_info:
      ps = FileWidget.PixelSize(info)
      img = BinaryImage(info)
      #dat = MeasureData(img, ps, self.psmax, self.measure_info)
      #self.features.append(dat)
      self.Progress.emit(inc)
    self.finished.emit()

"""
 ML Data Dialog
"""
class MLDataDialog(QtWidgets.QDialog):
  def __init__(self, parent):
    QtWidgets.QDialog.__init__(self, parent)
    self.setWindowTitle("ML Data")
    self.parent = parent
    hbox = QtWidgets.QHBoxLayout(self)
    vbox = QtWidgets.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = QtWidgets.QGraphicsView()
    self.scene = QtWidgets.QGraphicsScene()
    self.viewer.setScene(self.scene)
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.scene.addWidget(self.canvas)
    hbox.addWidget(self.viewer)
    self.treeview = QtWidgets.QTreeView()
    vbox.addWidget(self.treeview)
    self.treemodel = QtGui.QStandardItemModel()
    self.treemodel.setHorizontalHeaderLabels(['Keys', 'Values', 'Remarks'])
    self.treeview.setModel(self.treemodel)
    self.treeview.header().setStretchLastSection(False)
    self.treeitems = []
#    self.info = self.initMeasureInfo()
#    self.setMeasureInfo()
    #self.buttonb = QtGui.QDialogButtonBox()
    #vbox.addWidget(self.buttonb)
    #self.buttonb.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
    #self.buttonb.accepted.connect(self.Accept)
    #self.buttonb.rejected.connect(self.reject)
    
    #self.parent = parent
    #self.insitu = True
    #self.data = None
    #self.stat = None
    #self.psmin = None
    #self.afnd = None
    #self.measure = MeasureEachThread()
    #self.measure.finished.connect(self.measureFinish)
    #self.measure.Progress.connect(self.measureProgress)



    #vbox = QtGui.QVBoxLayout(self)
    #btn_mes = QtWidgets.QPushButton('Measure')
    #btn_mes.clicked[bool].connect(self.measureEach)
    #vbox.addWidget(btn_mes)




  def measureEach(self):
    if len(self.image_info) >= 1:
      self.pdlg = QtWidgets.QProgressDialog(self)
      self.pdlg.setWindowTitle("Measuring Each Size ...")
      self.pdlg.canceled.connect(self.measureCancel)
      self.pdlg1.setValue(0)
      self.measure_th.setup(self.image_info, self.psmax, self.measure_info)
      self.measure_th.start()

  def measureFinish(self):
    self.pdlg1.hide()
    self.features = self.measure_th.features
    self.labels = self.getLabels()

  def measureCancel(self):
    self.measure_th.terminate()

  def measureProgress(self, inc):
    val = self.pdlg1.value()
    self.pdlg1.setValue(val+inc)



"""
Scale Image Dialog
"""
class ScaleImageDialog(QtWidgets.QDialog):
  def __init__(self, parent):
    QtWidgets.QDialog.__init__(self, parent)
    self.setWindowTitle("Scale Image")
    self.parent = parent
    self.insitu = True
    self.plotimg = None
    self.plotps = None
    hbox = QtWidgets.QHBoxLayout(self)
    vbox = QtWidgets.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = QtWidgets.QGraphicsView()
    self.scene = QtWidgets.QGraphicsScene()
    self.viewer.setScene(self.scene)
    self.figure = Figure()
    self.figure.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)
    self.canvas = FigureCanvas(self.figure)
    self.scene.addWidget(self.canvas)
    hbox.addWidget(self.viewer)
    self.check1 = QtWidgets.QCheckBox('Draw Scale')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawImage)
    vbox.addWidget(self.check1)
    vbox.addWidget(QtWidgets.QLabel('Scale :'))
    self.line1 = QtWidgets.QLineEdit()
    self.line1.setValidator(QtGui.QDoubleValidator())
    self.line1.textChanged[str].connect(self.drawImage)
    self.line1.setText('100')
    vbox.addWidget(self.line1)
    vbox.addWidget(QtWidgets.QLabel('Font Size :'))
    self.spin1 = QtWidgets.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(100)
    self.spin1.setValue(24)
    self.spin1.valueChanged.connect(self.drawImage)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtWidgets.QLabel('Color :'))
    self.combo1 = QtWidgets.QComboBox()
    self.combo1.addItem('Black')
    self.combo1.addItem('White')
    self.combo1.addItem('Red')
    self.combo1.addItem('Green')
    self.combo1.addItem('Blue')
    self.combo1.addItem('Cyan')
    self.combo1.addItem('Magenta')
    self.combo1.addItem('Yellow')
    self.combo1.currentIndexChanged.connect(self.drawImage)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtWidgets.QLabel('DPI :'))
    self.spin2 = QtWidgets.QSpinBox()
    self.spin2.setMinimum(10)
    self.spin2.setMaximum(3000)
    self.spin2.setValue(100)
    vbox.addWidget(self.spin2)
    vbox.addStretch()
    hbox1 = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button1 = QtWidgets.QPushButton('Save Image')
    self.button1.clicked[bool].connect(self.saveImage)
    hbox1.addWidget(self.button1)
    self.button2 = QtWidgets.QPushButton('Close')
    self.button2.clicked[bool].connect(self.close)
    hbox1.addWidget(self.button2)

  def setInfo(self, info):
    self.insitu = False
    self.check1.setChecked(info['DrawScale'])
    self.line1.setText(info['Scale'])
    self.spin1.setValue(info['FontSize'])
    self.combo1.setCurrentIndex(info['Color'])
    self.insitu = True
    self.spin2.setValue(info['DPI'])

  def getInfo(self):
    info = {}
    info['DrawScale'] = self.check1.isChecked()
    info['Scale'] = str(self.line1.text())
    info['FontSize'] = self.spin1.value()
    info['Color'] = self.combo1.currentIndex()
    info['DPI'] = self.spin2.value()
    return info

  def setImage(self, img, info):
    if len(img.shape) == 2:
      self.plotimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:  
      self.plotimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w = 8.0
    h = w * img.shape[0] / img.shape[1]
    self.figure.set_size_inches(w, h)
    self.plotps = FileWidget.PixelSize(info)

  def drawScale(self, ax):
    try:
      sc = float(self.line1.text())
    except ValueError:
      sc = 0
    clist = ['k', 'w', 'r', 'g', 'b', 'c', 'm', 'y']
    cl = clist[self.combo1.currentIndex()]
    ll = sc / self.plotps[0]
    w = self.plotimg.shape[1]
    h = self.plotimg.shape[0]
    y = 0.975 * h
    sx = 0.975 * w - ll
    ex = 0.975 * w
    ax.hlines(y, sx, ex, lw=3, color=cl)
    txt = '%s%s' % (str(self.line1.text()), PlotUnitText(self.plotps[1]))
    txtc = len('%s%s' % (str(self.line1.text()), self.plotps[1]))
    fs = self.spin1.value()
    tw = 0.00124 * w * txtc * fs
    ts = sx + ll / 2 - tw / 2
    ax.text(ts, 0.96 * h, txt, fontsize=fs, color=cl)

  def drawImage(self):
    if self.plotimg is None or self.insitu == False:
      return
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1,frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(self.plotimg)
    if self.check1.isChecked():
      self.drawScale(ax)
    self.canvas.draw()

  def saveImage(self):
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Figure', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')[0]
    if fname:
      self.figure.savefig(str(fname), dpi=self.spin2.value())

"""
MainWindow
"""    
class MainWindow(QtWidgets.QMainWindow):
  def __init__(self, parent=None):
    QtWidgets.QWidget.__init__(self, parent)
    self.setWindowTitle(self.tr("MPImage"))
    self.image_info = []
    self.image_index = -1
    self.current_file = None
    splitter = QtWidgets.QSplitter()
    self.setCentralWidget(splitter)
    vsplitter = QtWidgets.QSplitter()
    vsplitter.setOrientation(QtCore.Qt.Vertical)
    splitter.addWidget(vsplitter)
    self.mainMenuBar()
    self.mainToolBar()
    self.tab = QtWidgets.QTabWidget()
    self.tab.currentChanged[int].connect(self.sceneChanged)
    vsplitter.addWidget(self.tab)
    imglist = QtWidgets.QWidget()
    vsplitter.addWidget(imglist)
    vbox = QtWidgets.QVBoxLayout(imglist)
    self.image_list = QtWidgets.QListWidget()
    self.image_list.currentRowChanged.connect(self.imageChanged)
    vbox.addWidget(self.image_list)
    hbox = QtWidgets.QHBoxLayout()
    vbox.addLayout(hbox)
    button1 = QtWidgets.QPushButton('Add Image')
    button1.clicked[bool].connect(self.addImage)
    hbox.addWidget(button1)
    button2 = QtWidgets.QPushButton('Delete Image')
    button2.clicked[bool].connect(self.deleteImage)
    hbox.addWidget(button2)
    #self.tab.setMinimumWidth(100)
    self.tab.setMaximumWidth(330)
    self.viewer = GraphicsView()
    self.scene = ImageScene()
    self.viewer.setScene(self.scene)
    splitter.addWidget(self.viewer)
    self.file = FileWidget(self)
    self.scene.measurePixel[float].connect(self.file.measurePixel)
    self.file.imageChanged.connect(self.scene.setImage)
    self.file.roiChanged.connect(self.scene.drawROI)
    self.tab.addTab(self.file, 'Src')
    self.cont = ContrastWidget(self)
    self.cont.imageChanged.connect(self.scene.setImage)
    self.tab.addTab(self.cont, 'Cont')
    self.filter = FilterWidget(self)
    self.filter.imageChanged.connect(self.scene.setImage)
    self.tab.addTab(self.filter, 'Fil')
    self.threshold = ThresholdWidget(self)
    self.threshold.imageChanged.connect(self.scene.setImage)
    self.tab.addTab(self.threshold, 'Thresh')
    self.morphology = MorphologyWidget(self)
    self.morphology.imageChanged.connect(self.scene.setImage)
    self.tab.addTab(self.morphology, 'Mor')
    self.modify = ModifyWidget(self)
    self.modify.imageChanged.connect(self.scene.setImage)
    self.modify.mouseModeChanged.connect(self.scene.setMouseMode)
    self.scene.addLine[float, float, float, float].connect(self.modify.addLine)
    self.scene.addRect[float, float, float, float].connect(self.modify.addRect)
    self.scene.clearRect[float, float, float, float].connect(self.modify.clearRect)
    self.tab.addTab(self.modify, 'Mod')
    self.imfp = IMFPDialog(self)
    self.ln2d = LN2DDialog(self)
    self.size = SizeDialog(self)
    self.mldata = MLDataDialog(self)
    self.scale = ScaleImageDialog(self)

  def sceneChanged(self):
    self.scene.clearAll()
    self.scene.mouseMode = self.scene.mouseNone
    if self.tab.currentIndex() == 0:
      self.file.setImage()
      self.scene.mouseMode = self.scene.mouseMeasure
    elif self.tab.currentIndex() == 1:
      self.cont.src_img = self.file.Process(self.file.org_img, self.file.getInfo())
      self.cont.setImage()
    elif self.tab.currentIndex() == 2:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      self.filter.src_img = self.cont.Process(roi_img, self.cont.getInfo())
      self.filter.setImage()
    elif self.tab.currentIndex() == 3:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      cont_img = self.cont.Process(roi_img, self.cont.getInfo())
      self.threshold.src_img = self.filter.Process(cont_img, self.filter.getInfo())
      self.threshold.setImage()
    elif self.tab.currentIndex() == 4:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      cont_img = self.cont.Process(roi_img, self.cont.getInfo())
      filter_img = self.filter.Process(cont_img, self.filter.getInfo())
      self.morphology.src_img = self.threshold.Process(filter_img, self.threshold.getInfo())
      self.morphology.setImage()
    elif self.tab.currentIndex() == 5:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      cont_img = self.cont.Process(roi_img, self.cont.getInfo())
      filter_img = self.filter.Process(cont_img, self.filter.getInfo())
      thresh_img = self.threshold.Process(filter_img, self.threshold.getInfo())
      self.modify.src_img = self.morphology.Process(thresh_img, self.morphology.getInfo())
      if roi_img is not None:
        self.modify.org_img = roi_img.copy()
      self.modify.setImage()
      self.modify.setMouseMode()

  def mainMenuBar(self):
    menubar = QtWidgets.QMenuBar(self)
    self.setMenuBar(menubar)
    file_menu = QtWidgets.QMenu('File', self) 
    file_menu.addAction('Open', self.fileOpen)
    file_menu.addAction('Save', self.fileSave)
    file_menu.addAction('Save As', self.fileSaveAs)
    file_menu.addAction('Clear All', self.clearAllMsg)
    file_menu.addAction('Save Image', self.saveImage)
    file_menu.addAction('Quit', self.close)   
    menubar.addMenu(file_menu)
    mes_menu = QtWidgets.QMenu('Measure', self)
    mes_menu.addAction('IMFP', self.measureIMFP)
    mes_menu.addAction('LN2D', self.measureLN2D)
    mes_menu.addAction('Size', self.measureSize)
    #mes_menu.addAction('ML Data', self.measureMLData)
    menubar.addMenu(mes_menu)
    misc_menu = QtWidgets.QMenu('Misc', self)
    misc_menu.addAction('Scale Image', self.miscScaleImage)
    menubar.addMenu(misc_menu)

  def clearAllMsg(self):
    msgbox = QtWidgets.QMessageBox(self)
    msgbox.setWindowTitle('Clear All')
    msgbox.setText('Do you want to clear all data ?')
    msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    msgbox.setDefaultButton(QtWidgets.QMessageBox.No)
    if msgbox.exec_() == QtWidgets.QMessageBox.Yes:
      self.clearAll()

  def clearAll(self):
    self.image_info = []
    self.image_index = -1
    self.imfp.clearFreq()
    self.ln2d.clearFreq()
    self.size.clearData()
    self.image_list.clear()

  def checkImageInfo(self, fname, image_info):
    missing = []
    for i in range(len(image_info)):
      img_fname = image_info[i]['FileName']
      img_sname = QtCore.QFileInfo(img_fname).fileName()
      if not QtCore.QFileInfo(img_fname).exists():
        img_nname = QtCore.QFileInfo(fname).path() + '/' + img_sname
        if QtCore.QFileInfo(img_nname).exists():
          image_info[i]['FileName'] = str(img_nname)
        else:
          msgbox = QtWidgets.QMessageBox(self)
          msgbox.setText("Can't find a image file, " + img_sname + '. Do you want to continue ?')
          msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
          msgbox.setDefaultButton(QtWidgets.QMessageBox.No)
          if msgbox.exec_() == QtWidgets.QMessageBox.Yes:
            missing.append(image_info[i])
          else:
            return []
    if len(missing) > 0:
      for info in missing:
         image_info.remove(info)
    return image_info

  def fileOpen(self):
    fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open', filter='JSON Files (*.json);;All Files (*.*)')[0]
    if fname[0]:
      fin = open(fname, 'r')
      data = json.load(fin)      
      self.clearAll()
      self.image_info = self.checkImageInfo(fname, data['ImageInfo'])
      for info in self.image_info:
        self.image_list.addItem(QtCore.QFileInfo(info['FileName']).fileName())
      self.image_list.setCurrentRow(self.image_list.count() - 1)
      self.imfp.setInfo(data['IMFPInfo'])
      self.ln2d.setInfo(data['LN2DInfo'])
      self.size.setInfo(data['SizeInfo'])
      self.scale.setInfo(data['ScaleInfo'])
      self.current_file = fname
      wt = 'MPImage - %s' % str(QtCore.QFileInfo(fname).fileName())
      self.setWindowTitle(self.tr(wt))
      
  def fileSave(self):
    if self.current_file != None:
      fout = open(self.current_file, 'w')
      data = {}
      self.updateImageInfo()
      data['ImageInfo'] = self.image_info
      data['IMFPInfo'] = self.imfp.getInfo()
      data['LN2DInfo'] = self.ln2d.getInfo()
      data['SizeInfo'] = self.size.getInfo()
      data['ScaleInfo'] = self.scale.getInfo()
      json.dump(data, fout, indent=2, sort_keys=True)
      fout.close()
    else:
      self.fileSaveAs()

  def fileSaveAs(self):
    fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save', filter='JSON Files (*.json);;All Files (*.*)')[0]
    if fname:
      self.current_file = fname
      wt = 'MPImage - %s' % str(QtCore.QFileInfo(fname).fileName())
      self.setWindowTitle(self.tr(wt))
      self.fileSave()

  def saveImage(self):
    if self.scene.cvimg is not None:
      fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save image', filter='Image Files (*.jpg *.png *.bmp *.ppm);;All Files (*.*)')[0]
      if fname:
        cv2.imwrite(str(fname), self.scene.cvimg)
        
  def measureIMFP(self):
    self.updateImageInfo()
    self.imfp.exec_()

  def measureLN2D(self):
    self.updateImageInfo()
    self.ln2d.exec_()

  def measureSize(self):
    self.updateImageInfo()
    self.size.exec_()

  def measureMLData(self):
    self.updateImageInfo()
    self.each.exec_()

  def miscScaleImage(self):
    if self.scene.cvimg is not None:
      self.updateImageInfo()
      self.scale.setImage(self.scene.cvimg, self.image_info[self.image_index])
      self.scale.drawImage()
      self.scale.exec_()

  def mainToolBar(self):
    toolbar = QtWidgets.QToolBar(self)
    self.addToolBar(toolbar)
    self.scale_label = QtWidgets.QLabel('100%')
    self.scale_label.setFixedWidth(30)
    toolbar.addWidget(self.scale_label)        
    self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    self.scale_slider.setFixedWidth(200)
    self.scale_slider.setMinimum(5)
    self.scale_slider.setMaximum(200)
    self.scale_slider.setValue(100)
    self.scale_slider.valueChanged[int].connect(self.changeScale)
    toolbar.addWidget(self.scale_slider)
    button3 = QtWidgets.QPushButton('Fit')
    button3.setFixedWidth(35)
    button3.clicked[bool].connect(self.fitScale)
    toolbar.addWidget(button3)    

  def addImage(self):
    fnames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add Image', filter='Image Files (*.jpg *.png *.tif *.bmp);;All Files (*.*)')
    if len(fnames) > 0:
      for fname in fnames[0]:
        print(fname)
        info = {"FileName":str(fname)}
        info.update(self.file.getInfo(fname))
        info.update(self.cont.getInfo())
        info.update(self.filter.getInfo())
        info.update(self.threshold.getInfo())
        info.update(self.morphology.getInfo())
        info.update(self.modify.getInfo(False))
        self.image_info.append(info)
        self.image_list.addItem(QtCore.QFileInfo(fname).fileName())
      self.image_list.setCurrentRow(self.image_list.count() - 1)

  def deleteImage(self):
    msgbox = QtWidgets.QMessageBox(self)
    item = self.image_list.currentItem()
    msgbox.setText('Do you want to delete %s ?' % item.text())
    msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    msgbox.setDefaultButton(QtWidgets.QMessageBox.No)
    if msgbox.exec_() == QtWidgets.QMessageBox.Yes:
      row = self.image_list.currentRow()
      self.image_info.pop(row)
      self.image_list.clear()
      for info in self.image_info:
        self.image_list.addItem(QtCore.QFileInfo(info['FileName']).fileName())
      nrow = row - 1
      if nrow < 0:
        nrow = 0
      self.image_index = -1
      self.image_list.setCurrentRow(nrow)

  def updateImageInfo(self):
    index = self.image_index
    if index >= 0 and index < len(self.image_info): 
      self.image_info[index].update(self.file.getInfo())
      self.image_info[index].update(self.cont.getInfo())
      self.image_info[index].update(self.filter.getInfo())
      self.image_info[index].update(self.threshold.getInfo())
      self.image_info[index].update(self.morphology.getInfo())
      self.image_info[index].update(self.modify.getInfo())   

  def imageChanged(self):
    self.updateImageInfo()
    row = self.image_list.currentRow()
    if row >= 0:
      info = self.image_info[row]
      self.file.setInfo(info)
      self.cont.setInfo(info)
      self.filter.setInfo(info)
      self.threshold.setInfo(info)
      self.morphology.setInfo(info)
      self.modify.setInfo(info)
      self.file.org_img = cv2.imread(info['FileName'], 1)
      self.sceneChanged()
      self.image_index = row
    else:
      self.file.clearFile()
      self.morphology.clearLabel()
      self.scene.clearAll()

  def changeScale(self):
    val = self.scale_slider.value()
    self.scale_label.setText(str(val) + '%')
    self.scene.setScale(val / 100.0)
    
  def fitScale(self):
    s = int(self.scene.calcFitScale() * 100.0)
    self.scale_slider.setValue(s)

if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)
  window = MainWindow()
  window.show()
  sys.exit(app.exec_())