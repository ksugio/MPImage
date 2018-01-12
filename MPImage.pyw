#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 28 12:04:57 2016

@author: Kenjiro Sugio
"""

import MPImfp
import MPLn23d
import cv2
import sys
import random
import numpy as np
import json
import math as m
from PyQt4 import QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

"""
Image Scene
"""  
class ImageScene(QtGui.QGraphicsScene):
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
    self.pixmap = None
    self.imageItem = None
    self.roiItem = None
    self.scale = 1.0
    self.startPos = None
    self.mouseMode = self.mouseNone

  def setImage(self, cvimg):
    if len(cvimg.shape) == 2:
      height, width = cvimg.shape
      qimg = QtGui.QImage(cvimg.data, width, height, width, QtGui.QImage.Format_Indexed8)
    elif len(cvimg.shape) == 3:  
      height, width, dim = cvimg.shape      
      qimg = QtGui.QImage(cvimg.data, width, height, dim*width, QtGui.QImage.Format_RGB888)
      qimg = qimg.rgbSwapped()
    self.pixmap = QtGui.QPixmap.fromImage(qimg)
    if self.imageItem:
      self.removeItem(self.imageItem)
    self.imageItem = QtGui.QGraphicsPixmapItem(self.pixmap)
    self.addItem(self.imageItem)
    self.__Scale()

  def clearImage(self):
    if self.imageItem:
      self.removeItem(self.imageItem)
      self.imageItem = None

  def __Scale(self):
    if self.imageItem:
      self.imageItem.setScale(self.scale)
      w = self.scale * self.pixmap.width()
      h = self.scale * self.pixmap.height()
      self.setSceneRect(0, 0, w, h)
    if self.roiItem:
      self.roiItem.setScale(self.scale)

  def setScale(self, scale):
    self.scale = scale
    self.__Scale()

  def calcFitScale(self):
    if self.pixmap != None:
      view = self.views()[0]      
      sw = float(view.width()) / float(self.pixmap.width())
      sh = float(view.height()) / float(self.pixmap.height())
      if sw < sh:
        return sw
      else:
        return sh
    else:
      return 1.0

  def drawROI(self, x, y, w, h):
    if self.roiItem:
      self.removeItem(self.roiItem)    
    self.roiItem = QtGui.QGraphicsRectItem()
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
        self.line_item = QtGui.QGraphicsLineItem()
        pen = QtGui.QPen(QtGui.QColor(0,255,0))
        self.line_item.setPen(pen)
        self.addItem(self.line_item)
      elif self.mouseMode == self.mouseRect:
        self.rect_item = QtGui.QGraphicsRectItem()
        pen = QtGui.QPen(QtGui.QColor(0,255,0))
        self.rect_item.setPen(pen)
        self.addItem(self.rect_item)
      elif self.mouseMode == self.mouseClear:
        self.rect_item = QtGui.QGraphicsRectItem()
        pen = QtGui.QPen(QtGui.QColor(0,0,255))
        self.rect_item.setPen(pen)
        self.addItem(self.rect_item)        
      elif self.mouseMode == self.mouseMeasure:
        self.line_item = QtGui.QGraphicsLineItem()
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
        self.rect_item.setRect(start.x(), start.y(), cur.x()-start.x(), cur.y()-start.y())
      elif self.mouseMode == self.mouseClear:  
        self.rect_item.setRect(start.x(), start.y(), cur.x()-start.x(), cur.y()-start.y())        
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
        dis = m.sqrt(dx*dx + dy*dy)
        self.measurePixel.emit(dis)
      self.startPos = None
    super(ImageScene, self).mouseReleaseEvent(event)

"""
Graphics View
"""  
class GraphicsView(QtGui.QGraphicsView):
  def __init__(self):
    super(GraphicsView, self).__init__()
    self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
    self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform | QtGui.QPainter.TextAntialiasing)

  def sizeHint(self):
    return QtCore.QSize(1024, 768)

"""
File Widget
"""  
class FileWidget(QtGui.QWidget):
  def __init__(self, parent, scene):
    super(FileWidget, self).__init__(parent)
    self.scene = scene
    self.org_img = None
    self.insitu = True
    vbox = QtGui.QVBoxLayout(self)
    self.name_label = QtGui.QLabel('Name :')
    vbox.addWidget(self.name_label)
    self.size_label = QtGui.QLabel('Size :')
    vbox.addWidget(self.size_label)
    vbox.addWidget(QtGui.QLabel('Resize :'))
    hbox = QtGui.QHBoxLayout()
    vbox.addLayout(hbox)
    self.spin5 = QtGui.QSpinBox()
    self.spin5.valueChanged[int].connect(self.changeWidth)
    hbox.addWidget(self.spin5)
    self.spin6 = QtGui.QSpinBox()
    self.spin6.valueChanged[int].connect(self.changeHeight)
    hbox.addWidget(self.spin6)    
    vbox.addWidget(QtGui.QLabel('ROI :'))
    glay = QtGui.QGridLayout()
    vbox.addLayout(glay)
    self.spin1 = QtGui.QSpinBox()
    self.spin1.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin1, 0, 0)
    self.spin2 = QtGui.QSpinBox()
    self.spin2.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin2, 0, 1)
    self.spin3 = QtGui.QSpinBox()
    self.spin3.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin3, 1, 0)
    self.spin4 = QtGui.QSpinBox()
    self.spin4.valueChanged[int].connect(self.changeROI)
    glay.addWidget(self.spin4, 1, 1)
    button2 = QtGui.QPushButton('Reset ROI')
    button2.clicked[bool].connect(self.resetROI)
    vbox.addWidget(button2)
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
    self.insitu = True
    
  def setInfo(self, info):
    self.insitu = False
    roi = info['FileROI']
    size = info['FileSize']
    resize = info['FileResize']
    fname = info['FileName']
    filename = QtCore.QFileInfo(fname).fileName()
    self.name_label.setText('Name : ' + filename)    
    self.size_label.setText('Size : ' + str(size[0]) + ' x '  + str(size[1]))
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
    self.spin5.setMaximum(10*size[0]) 
    self.spin5.setValue(resize[0])
    self.spin6.setMinimum(1)
    self.spin6.setMaximum(10*size[1])
    self.spin6.setValue(resize[1])
    self.insitu = True
    
  def getInfo(self, fname=None):
    if fname == None:
      if self.org_img is not None:
        height, width, dim = self.org_img.shape
        size = [width, height]
      else:
        size = [0, 0]
      resize = [self.spin5.value(), self.spin6.value()]
      roi = [self.spin1.value(), self.spin2.value(), self.spin3.value(), self.spin4.value()]
      return {'FileSize':size, 'FileResize':resize, 'FileROI':roi}
    else:
      img = cv2.imread(str(fname), 1)
      height, width, dim = img.shape
      size = [width, height]
      roi = [0, 0, width, height]
      return {'FileSize':size, 'FileResize':size, 'FileROI':roi}

  def changeROI(self):
    if self.insitu:
      sx = self.spin1.value()
      sy = self.spin2.value()
      ex = self.spin3.value()
      ey = self.spin4.value()
      self.scene.drawROI(sx, sy, ex-sx, ey-sy)
 
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
        self.scene.setImage(self.org_img)
      else:
        res_img = cv2.resize(self.org_img, (self.spin5.value(), self.spin6.value()))
        self.scene.setImage(res_img)
      sx = self.spin1.value()
      sy = self.spin2.value()
      ex = self.spin3.value()
      ey = self.spin4.value()
      self.scene.drawROI(sx, sy, ex-sx, ey-sy)

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

"""
Filter Widget
"""  
class FilterWidget(QtGui.QWidget):
  def __init__(self, parent, scene):
    super(FilterWidget, self).__init__(parent)
    self.scene = scene
    self.src_img = None
    self.insitu = True
    vbox = QtGui.QVBoxLayout(self)
    vbox.addWidget(QtGui.QLabel('Type :'))
    self.combo1 = QtGui.QComboBox()
    self.combo1.addItem('None')
    self.combo1.addItem('Blur')
    self.combo1.addItem('Gaussian')
    self.combo1.addItem('Median')
    self.combo1.addItem('Bilateral')
    self.combo1.currentIndexChanged[int].connect(self.filterChanged)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtGui.QLabel('Size :'))   
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(99)
    self.spin1.setValue(1)
    self.spin1.setSingleStep(2)
    self.spin1.setEnabled(False)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtGui.QLabel('Sigma0 :'))   
    self.spin2 = QtGui.QSpinBox()
    self.spin2.setMinimum(0)
    self.spin2.setMaximum(300)
    self.spin2.setValue(0)
    self.spin2.setEnabled(False)
    self.spin2.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin2)   
    vbox.addWidget(QtGui.QLabel('Sigma1 :'))   
    self.spin3 = QtGui.QSpinBox()
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
        self.scene.setImage(dst_img)

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
class ThresholdWidget(QtGui.QWidget):
  def __init__(self, parent, scene):
    super(ThresholdWidget, self).__init__(parent)
    self.scene = scene
    self.src_img = None
    self.dst_img = None
    self.insitu = True
    vbox = QtGui.QVBoxLayout(self)
    vbox.addWidget(QtGui.QLabel('Type :'))
    self.combo1 = QtGui.QComboBox()
    self.combo1.addItem('Simple')
    self.combo1.addItem('Otsu')
    self.combo1.addItem('Adaptive Mean')
    self.combo1.addItem('Adaptive Gauss')
    self.combo1.currentIndexChanged[int].connect(self.methodChanged)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtGui.QLabel('Threshold :'))   
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(255)
    self.spin1.setValue(127)
    self.spin1.setEnabled(True)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtGui.QLabel('Adaptive Block Size :'))   
    self.spin2 = QtGui.QSpinBox()   
    self.spin2.setMinimum(3)
    self.spin2.setMaximum(999)
    self.spin2.setValue(123)   
    self.spin2.setSingleStep(2)
    self.spin2.setEnabled(False)
    self.spin2.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin2)   
    vbox.addWidget(QtGui.QLabel('Adaptive Parm :'))   
    self.spin3 = QtGui.QSpinBox()
    self.spin3.setMinimum(-128)
    self.spin3.setMaximum(128)
    self.spin3.setValue(0)
    self.spin3.setEnabled(False)
    self.spin3.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin3)
    self.check1 = QtGui.QCheckBox('Invert')
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
        self.scene.setImage(dst_img)

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
      ret, dst_img = cv2.threshold(src_img, 0, 255, sty+cv2.THRESH_OTSU)
    elif ttype == 2:
      dst_img = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, sty, bsize, parm)
    elif ttype == 3:
      dst_img = cv2.adaptiveThreshold(src_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, sty, bsize, parm)
    return dst_img          

"""
Morphology Widget
"""  
class MorphologyWidget(QtGui.QWidget):
  def __init__(self, parent, scene):
    super(MorphologyWidget, self).__init__(parent)
    self.scene = scene
    self.src_img = None
    self.insitu = True
    vbox = QtGui.QVBoxLayout(self)
    vbox.addWidget(QtGui.QLabel('Type :'))
    self.combo1 = QtGui.QComboBox()
    self.combo1.addItem('Opening')
    self.combo1.addItem('Closing')
    self.combo1.currentIndexChanged[int].connect(self.setImage)
    vbox.addWidget(self.combo1)  
    vbox.addWidget(QtGui.QLabel('Iterations :'))
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(32)
    self.spin1.setValue(0)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    vbox.addStretch() 
    self.label1 = QtGui.QLabel('Black :')
    vbox.addWidget(self.label1)
    self.label2 = QtGui.QLabel('White :')
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
        self.scene.setImage(dst_img)
        hist = cv2.calcHist([dst_img], [0], None, [256], [0,256])
        tot = hist[0][0]+hist[255][0]
        self.label1.setText('Black : ' + str(100.0*hist[0][0]/tot) + ' %')
        self.label2.setText('White : ' + str(100.0*hist[255][0]/tot) + ' %')

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
class ModifyWidget(QtGui.QWidget):
  def __init__(self, parent, scene):
    super(ModifyWidget, self).__init__(parent)
    self.scene = scene
    self.scene.addLine[float, float, float, float].connect(self.addLine)
    self.scene.addRect[float, float, float, float].connect(self.addRect)
    self.scene.clearRect[float, float, float, float].connect(self.clearRect)    
    self.src_img = None
    self.org_img = None
    self.mod_objects = []
    self.insitu = True
    vbox = QtGui.QVBoxLayout(self)
    vbox.addWidget(QtGui.QLabel('Source weight :'))
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(0)
    self.spin1.setMaximum(255)
    self.spin1.setValue(255)
    self.spin1.valueChanged[int].connect(self.setImage)
    vbox.addWidget(self.spin1)
    self.check1 = QtGui.QCheckBox('Contour draw')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check1)
    self.check2 = QtGui.QCheckBox('Center draw')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check2)
    self.check4 = QtGui.QCheckBox('BoundRect draw')
    self.check4.setChecked(True)
    self.check4.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check4)    
    self.check3 = QtGui.QCheckBox('Modify draw')
    self.check3.setChecked(True)
    self.check3.stateChanged[int].connect(self.setImage)
    vbox.addWidget(self.check3)
    vbox.addWidget(QtGui.QLabel('Modify color :'))
    hbox1 = QtGui.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.group1 = QtGui.QButtonGroup()
    button1 = QtGui.QPushButton('Black')
    button1.setCheckable(True)
    button1.setChecked(True)
    self.group1.addButton(button1, 0)
    hbox1.addWidget(button1)
    button2 = QtGui.QPushButton('White')
    button2.setCheckable(True)
    self.group1.addButton(button2, 1)
    hbox1.addWidget(button2)    
    vbox.addWidget(QtGui.QLabel('Modify mode :'))
    hbox2 = QtGui.QHBoxLayout()
    vbox.addLayout(hbox2)
    self.group2 = QtGui.QButtonGroup()
    self.group2.buttonClicked.connect(self.setMouseMode)
    button3 = QtGui.QPushButton('Line')
    button3.setCheckable(True)
    button3.setChecked(True)
    self.group2.addButton(button3, 0)
    hbox2.addWidget(button3)
    button4 = QtGui.QPushButton('Rect')
    button4.setCheckable(True)
    self.group2.addButton(button4, 1)
    hbox2.addWidget(button4)
    button5 = QtGui.QPushButton('Clear')
    button5.setCheckable(True)
    self.group2.addButton(button5, 2)
    hbox2.addWidget(button5)   
    vbox.addWidget(QtGui.QLabel('Line width :'))
    self.spin2 = QtGui.QSpinBox()
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
        self.scene.mouseMode = self.scene.mouseLine
      elif self.group2.checkedId() == 1:
        self.scene.mouseMode = self.scene.mouseRect
      elif self.group2.checkedId() == 2:
        self.scene.mouseMode = self.scene.mouseClear  

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
        self.scene.setImage(dst_img)  

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
          x = int(mom['m10']/mom['m00'])
          y = int(mom['m01']/mom['m00'])
          cv2.line(draw_img, (x, y-3), (x, y+3), (0, 0, 255))
          cv2.line(draw_img, (x-3, y), (x+3, y), (0, 0, 255))
    if info['ModBoundRectDraw']:
      for cont in conts:
        rect = cv2.minAreaRect(cont)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (255, 0, 0), 1)
    if info['ModContourDraw']:
      cv2.drawContours(draw_img, conts, -1, (255,255,0), 1)
    return draw_img

"""
Graph Scene
"""
class GraphScene(QtGui.QGraphicsScene):
  def __init__(self, *argv, **keywords):
    super(GraphScene, self).__init__(*argv, **keywords)
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.canvas.setGeometry(0, 0, 640, 480)
    #self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
    self.canvas.updateGeometry()
    self.addWidget(self.canvas)

  def drawIMFP(self, freq, xlabel, ps, pixmax, rf_plot, show_stat):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    length = np.arange(freq.size) * ps
    if rf_plot:
      ax.plot(length, self.rFreq(freq), 'k-')
      ax.set_ylabel('Relative Frequency')
    else:
      ax.plot(length, freq, 'k-')
      ax.set_ylabel('Frequency')
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, pixmax*ps)
    if show_stat:
      stat = self.calcStat(freq, ps)
      ax.text(0.6, 0.95, 'NSample : ' + str(stat[0]), transform=ax.transAxes)
      ax.text(0.6, 0.9, 'Mean : ' + str(stat[1]), transform=ax.transAxes) 
      ax.text(0.6, 0.85, 'STD : ' + str(stat[3]), transform=ax.transAxes)
    self.canvas.draw()

  def drawSize(self, data, xlabel, bins, show_stat):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    ax.hist(data, bins=bins, color='white', rwidth=0.8)
    ax.set_ylabel('Frequency')
    ax.set_xlabel(xlabel)
    if show_stat:
      ax.text(0.6, 0.95, 'NSample : ' + str(data.shape[0]), transform=ax.transAxes)
      ax.text(0.6, 0.9, 'Mean : ' + str(data.mean()), transform=ax.transAxes) 
      ax.text(0.6, 0.85, 'STD : ' + str(data.std()), transform=ax.transAxes)
    self.canvas.draw()

  def drawLN2D(self, freq, xlabel, lnmax, rf_plot, show_stat, prob):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    if rf_plot:
      ax.bar(np.arange(freq.size)-0.4, self.rFreq(freq), color='white')
      if prob != None:
        ax.plot(prob[0], prob[1], 'k-')
      ax.set_ylabel('Relative Frequency')
    else:
      ax.bar(np.arange(freq.size)-0.4, freq, color='white')
      if prob != None:
        tot = np.sum(freq)
        ax.plot(prob[0], tot*prob[1], 'k-')      
      ax.set_ylabel('Frequency')
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.5, lnmax)
    if show_stat:
      stat = self.calcStat(freq)
      ax.text(0.6, 0.95, 'NSample : ' + str(stat[0]), transform=ax.transAxes)
      ax.text(0.6, 0.9, 'Av. : ' + str(stat[1]), transform=ax.transAxes)
      ax.text(0.6, 0.85, 'Var. : ' + str(stat[2]), transform=ax.transAxes)
      if prob != None:
        ax.text(0.6, 0.8, 'Ref. Av. : ' + str(prob[2]), transform=ax.transAxes)
        ax.text(0.6, 0.75, 'Ref. Var. : ' + str(prob[3]), transform=ax.transAxes)      
    self.canvas.draw()    

  def drawImage(self, img):
    self.figure.clf()
    ax = self.figure.add_subplot(1,1,1)
    ax.imshow(img)
    self.canvas.draw()

  def clearGraph(self):
    self.figure.clf()
    self.canvas.draw()
      
  def saveGraph(self, fname):
    self.figure.savefig(fname)

  @staticmethod
  def unitText(unit):
    if unit == 'um':
      return '$\mu$m'
    else:
      return unit

  @staticmethod
  def rFreq(freq):
    tot = np.sum(freq)
    return np.array(freq, dtype=np.float)/tot

  @staticmethod
  def calcStat(freq, step=None):
    tot = np.sum(freq)
    rf = np.array(freq, dtype=np.float)/tot
    if step == None:
      length = np.arange(freq.size)
    else:
      length = np.arange(freq.size)*step
    ave = np.sum(length*rf)
    var = np.sum(np.power(length-ave, 2)*rf)
    std = m.sqrt(var)
    return [tot, ave, var, std]

"""
Measure IMFP Thread
"""
class MeasureIMFPThread(QtCore.QThread):
  Progress = QtCore.pyqtSignal(int, str)  
  
  def __init__(self, parent=None):   
    super(MeasureIMFPThread, self).__init__(parent)
    self.mutex = QtCore.QMutex()
    self.nsample = 0
    self.seed = 0
    self.freq = None
    self.image_info = None

  def setup(self, nsample, seed, image_info, freq):
    self.nsample = nsample
    self.seed = seed
    self.image_info = image_info
    self.freq = freq

  def run(self):
    inc = 100/len(self.image_info)
    for info in self.image_info:
      fname = info['FileName']
      org_img = cv2.imread(fname, 1)
      roi_img = FileWidget.Process(org_img, info)
      filter_img = FilterWidget.Process(roi_img, info)
      thresh_img = ThresholdWidget.Process(filter_img, info)
      morphol_img = MorphologyWidget.Process(thresh_img, info)
      mod_img = ModifyWidget.ProcessMod(morphol_img, info)
      #cv2.imshow('IMFP measure', mod_img)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()
      MPImfp.measure(mod_img, 255, self.freq[0], self.nsample, self.seed, 0)
      self.seed = MPImfp.measure(mod_img, 255, self.freq[1], self.nsample, self.seed, 1)
      filename = QtCore.QFileInfo(fname).fileName()
      self.Progress.emit(inc, filename)
    self.finished.emit()

"""
IMFP Dialog
"""
class IMFPDialog(QtGui.QDialog):
  def __init__(self, parent):
    QtGui.QDialog.__init__(self, parent)
    self.setWindowTitle("IMFP")
    self.parent = parent
    self.insitu = True
    self.freq = None
    self.measure = MeasureIMFPThread()
    self.measure.finished.connect(self.measureFinish)
    self.measure.Progress.connect(self.measureProgress)
    hbox = QtGui.QHBoxLayout(self)
    vbox = QtGui.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = GraphicsView()
    self.viewer.setFixedSize(642, 482)
    self.scene = GraphScene()
    self.viewer.setScene(self.scene)
    hbox.addWidget(self.viewer)
    vbox.addWidget(QtGui.QLabel('NSample (x10000) :'))
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(1000)
    self.spin1.setValue(100)      
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtGui.QLabel('Pixel Max :'))
    self.spin2 = QtGui.QSpinBox()
    self.spin2.setMinimum(100)
    self.spin2.setMaximum(10000)
    self.spin2.setValue(5000)      
    vbox.addWidget(self.spin2)
    vbox.addWidget(QtGui.QLabel('Seed :'))
    self.line1 = QtGui.QLineEdit()
    seed = random.randint(1, 1000000000)
    self.line1.setText(str(seed))     
    vbox.addWidget(self.line1)
    self.button1 = QtGui.QPushButton('Measure')
    self.button1.clicked[bool].connect(self.measureIMFP)
    vbox.addWidget(self.button1)
    self.pbar1 = QtGui.QProgressBar()
    self.pbar1.setRange(0,100)
    self.pbar1.setValue(0)
    vbox.addWidget(self.pbar1)
    self.list1 = QtGui.QListWidget()
    vbox.addWidget(self.list1)
    vbox.addWidget(QtGui.QLabel('Type :'))
    self.combo1 = QtGui.QComboBox()
    self.combo1.addItem('Single')  
    self.combo1.addItem('Double')
    self.combo1.currentIndexChanged.connect(self.drawGraph)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtGui.QLabel('Plot Pixel Max :'))
    self.spin3 = QtGui.QSpinBox()
    self.spin3.setMinimum(10)
    self.spin3.setMaximum(10000)
    self.spin3.setValue(5000)
    self.spin3.valueChanged.connect(self.drawGraph)     
    vbox.addWidget(self.spin3)
    self.check1 = QtGui.QCheckBox('Relative Frequency')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check1)
    self.check2 = QtGui.QCheckBox('Show Statistics')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check2)    
    vbox.addStretch()
    hbox1 = QtGui.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button2 = QtGui.QPushButton('Save CSV')
    self.button2.clicked[bool].connect(self.saveCSV)
    hbox1.addWidget(self.button2)
    self.button3 = QtGui.QPushButton('Save Graph')
    self.button3.clicked[bool].connect(self.saveGraph)
    hbox1.addWidget(self.button3)
    self.button4 = QtGui.QPushButton('Close')
    self.button4.clicked[bool].connect(self.close)
    hbox1.addWidget(self.button4)

  def setInfo(self, info):
    self.spin1.setValue(info['NSample'])    
    self.spin2.setValue(info['PixelMax'])
    self.line1.setText(info['Seed'])
    self.insitu = False
    self.combo1.setCurrentIndex(info['Type'])
    self.spin3.setValue(info['PlotPixelMax'])
    self.check1.setChecked(info['RelativeFrequency'])
    self.check2.setChecked(info['ShowStatistics'])
    self.insitu = True
    self.list1.clear()
    for it in info['List']:    
      self.list1.addItem(it)
    if info['Freq'] != None:
      self.freq = np.array(info['Freq'], dtype=np.uint32)
    else:
      self.freq = None
    self.drawGraph()

  def getInfo(self):
    info = {}
    info['NSample'] = self.spin1.value()
    info['PixelMax'] = self.spin2.value()
    info['Seed'] = str(self.line1.text())
    info['Type'] = self.combo1.currentIndex()
    info['PlotPixelMax'] = self.spin3.value()
    info['RelativeFrequency'] = self.check1.isChecked()
    info['ShowStatistics'] = self.check2.isChecked()
    lis = []
    for i in range(self.list1.count()):    
      lis.append(str(self.list1.item(i).text()))
    info['List'] = lis
    if self.freq != None:
      info['Freq'] = self.freq.tolist()
    else:
      info['Freq'] = None  
    return info

  def measureIMFP(self):
    if self.parent.image_index >= 0:
      nsample = self.spin1.value() * 10000
      pixmax = self.spin2.value()
      seed = long(self.line1.text())
      self.freq = np.zeros((2, pixmax), dtype=np.uint32)
      self.list1.clear()
      self.button1.setEnabled(False) 
      self.button2.setEnabled(False)  
      self.button3.setEnabled(False)  
      self.button4.setEnabled(False)
      self.combo1.setEnabled(False)
      self.spin3.setEnabled(False)
      self.check1.setEnabled(False)
      self.check2.setEnabled(False)
      self.pbar1.setValue(0)
      self.measure.setup(nsample, seed, self.parent.image_info, self.freq)        
      self.measure.start()

  def measureProgress(self, inc, filename):
    val = self.pbar1.value()
    self.pbar1.setValue(val+inc)
    self.list1.addItem(filename)

  def measureFinish(self):
    self.button1.setEnabled(True) 
    self.button2.setEnabled(True)     
    self.button3.setEnabled(True) 
    self.button4.setEnabled(True)
    self.combo1.setEnabled(True)
    self.spin3.setEnabled(True)
    self.check1.setEnabled(True)
    self.check2.setEnabled(True)
    self.pbar1.setValue(100)
    self.drawGraph()

  def drawGraph(self):
    if self.freq is None or self.insitu == False:
      return
    ps = self.parent.pixelsize.data['PS']
    unit = self.parent.pixelsize.data['UNIT']
    if self.combo1.currentIndex() == 0:
      xlabel = 'Single length (' + self.scene.unitText(unit) + ')'
      self.scene.drawIMFP(self.freq[0], xlabel, ps, self.spin3.value(),\
        self.check1.isChecked(), self.check2.isChecked())
    elif self.combo1.currentIndex() == 1:
      xlabel = 'Double length (' + self.scene.unitText(unit) + ')'
      self.scene.drawIMFP(self.freq[1], xlabel, ps, self.spin3.value(),\
        self.check1.isChecked(), self.check2.isChecked())

  def showEvent(self, event):
    self.drawGraph()

  def saveCSV(self):    
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV Files (*.csv);;All Files (*.*)')
    if fname:
      fout = open(fname, 'w')  
      fout.write('Images,' + str(self.list1.count()) + '\n')
      for i in range(self.list1.count()):
        fout.write(self.list1.item(i).text() + '\n')
      psdata = self.parent.pixelsize.data
      fout.write('PixelSize,' + str(psdata['PS']) + ',' + psdata['UNIT'] + '\n')
      fout.write('Pixmax,' + str(len(self.freq[0])) + '\n')
      fout.write('Statistics, Total, Mean, STD\n')
      stat0 = self.scene.calcStat(self.freq[0], step=psdata['PS'])
      fout.write('Single' + ',' + str(stat0[0]) + ',' + str(stat0[1]) + ',' + str(stat0[3]) + '\n')
      stat1 = self.scene.calcStat(self.freq[1], step=psdata['PS'])
      fout.write('Double' + ',' + str(stat1[0]) + ',' + str(stat1[1]) + ',' + str(stat1[3]) + '\n')
      fout.write('Pixel, Length, SingleF, SingleRF, DoubleF, DoubleRF\n')
      rf0 = np.array(self.freq[0], dtype=np.float) / stat0[0]
      rf1 = np.array(self.freq[1], dtype=np.float) / stat1[0]
      for i in range(len(self.freq[0])):
        fout.write(str(i) + ',' + str(i*psdata['PS']) + ',' + str(self.freq[0,i]) + ',' + str(rf0[i]) + ','\
          + str(self.freq[1,i]) + ',' + str(rf1[i]) + '\n')

  def saveGraph(self):
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save Graph', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')
    if fname:
      self.scene.saveGraph(str(fname))    

  def clearFreq(self):
    self.freq = None
    self.pbar1.setValue(0)
    self.list1.clear()
    self.scene.clearGraph()

"""
Measure LN2D Thread
"""
class MeasureLN2DThread(QtCore.QThread):
  Progress = QtCore.pyqtSignal(int, str)  
  
  def __init__(self, parent=None):   
    super(MeasureLN2DThread, self).__init__(parent)
    self.mutex = QtCore.QMutex()
    self.nsample = 0
    self.freq = None
    self.image_info = None
    self.ln2d = None
    self.af = 0

  def setup(self, nsample, seed, image_info, freq):
    self.nsample = nsample
    self.image_info = image_info
    self.freq = freq
    self.ln2d = MPLn23d.ln2d_new(len(self.image_info))
    self.ln2d.seed = seed

  def run(self):
    inc = 100/len(self.image_info)
    for info in self.image_info:
      fname = info['FileName']
      org_img = cv2.imread(fname, 1)
      roi_img = FileWidget.Process(org_img, info)
      filter_img = FilterWidget.Process(roi_img, info)
      thresh_img = ThresholdWidget.Process(filter_img, info)
      morphol_img = MorphologyWidget.Process(thresh_img, info)
      mod_img = ModifyWidget.ProcessMod(morphol_img, info)      
      #cv2.imshow('IMFP measure', mod_img)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()
      conts, hier = cv2.findContours(mod_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      sid = self.ln2d.add_sec(len(conts), mod_img.shape[1], mod_img.shape[0])
      for cont in conts:
        mom = cv2.moments(cont)
        if mom['m00'] != 0:
          x = mom['m10']/mom['m00']
          y = mom['m01']/mom['m00']
          area = cv2.contourArea(cont)
          r = m.sqrt(area/m.pi)
          self.ln2d.add_gc(sid, x, y, r)
      filename = QtCore.QFileInfo(fname).fileName()
      self.Progress.emit(inc, filename)
    self.ln2d.measure_gc(self.freq[0])
    self.ln2d.measure_random(self.freq[1], self.nsample)
    self.af = self.ln2d.area_fraction()
    self.finished.emit()

"""
LN2D Dialog
"""
class LN2DDialog(QtGui.QDialog):
  def __init__(self, parent):
    QtGui.QDialog.__init__(self, parent)
    self.setWindowTitle("LN2D")
    self.parent = parent
    self.insitu = True
    self.freq = None
    self.measure = MeasureLN2DThread()
    self.measure.finished.connect(self.measureFinish)
    self.measure.Progress.connect(self.measureProgress)
    hbox = QtGui.QHBoxLayout(self)
    vbox = QtGui.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = GraphicsView()
    self.viewer.setFixedSize(642, 482)
    self.scene = GraphScene()
    self.viewer.setScene(self.scene)
    hbox.addWidget(self.viewer)
    vbox.addWidget(QtGui.QLabel('NSample (x10000) :'))
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(1000)
    self.spin1.setValue(100)      
    vbox.addWidget(self.spin1)
    vbox.addWidget(QtGui.QLabel('LN Max :'))
    self.spin2 = QtGui.QSpinBox()
    self.spin2.setMinimum(10)
    self.spin2.setMaximum(1000)
    self.spin2.setValue(100)
    vbox.addWidget(self.spin2)
    vbox.addWidget(QtGui.QLabel('Seed :'))
    self.line1 = QtGui.QLineEdit()
    seed = random.randint(1, 1000000000)
    self.line1.setText(str(seed))     
    vbox.addWidget(self.line1)
    self.button1 = QtGui.QPushButton('Measure')
    self.button1.clicked[bool].connect(self.measureLN2D)
    vbox.addWidget(self.button1)
    self.pbar1 = QtGui.QProgressBar()
    self.pbar1.setRange(0,100)
    self.pbar1.setValue(0)
    vbox.addWidget(self.pbar1)
    self.list1 = QtGui.QListWidget()
    vbox.addWidget(self.list1)
    vbox.addWidget(QtGui.QLabel('Type :'))
    self.combo1 = QtGui.QComboBox()
    self.combo1.addItem('Gravity center')  
    self.combo1.addItem('Random')
    self.combo1.currentIndexChanged.connect(self.drawGraph)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtGui.QLabel('Plot LN Max :'))
    self.spin3 = QtGui.QSpinBox()
    self.spin3.setMinimum(1)
    self.spin3.setMaximum(1000)
    self.spin3.setValue(30)
    self.spin3.valueChanged.connect(self.drawGraph)     
    vbox.addWidget(self.spin3)
    vbox.addWidget(QtGui.QLabel('Area fraction :'))
    self.spin4 = QtGui.QSpinBox()
    self.spin4.setMinimum(0)
    self.spin4.setMaximum(100)
    self.spin4.setValue(0)
    self.spin4.valueChanged.connect(self.drawGraph)         
    vbox.addWidget(self.spin4)
    self.check1 = QtGui.QCheckBox('Relative Frequency')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check1)
    self.check2 = QtGui.QCheckBox('Show Statistics')
    self.check2.setChecked(True)
    self.check2.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check2)
    self.check3 = QtGui.QCheckBox('Show Reference')
    self.check3.setChecked(True)
    self.check3.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check3)
    vbox.addStretch()
    hbox1 = QtGui.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button2 = QtGui.QPushButton('Save CSV')
    self.button2.clicked[bool].connect(self.saveCSV)
    hbox1.addWidget(self.button2)
    self.button3 = QtGui.QPushButton('Save Graph')
    self.button3.clicked[bool].connect(self.saveGraph)
    hbox1.addWidget(self.button3)
    self.button4 = QtGui.QPushButton('Close')
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
    self.list1.clear()
    for it in info['List']:    
      self.list1.addItem(it)
    if info['Freq'] != None:
      self.freq = np.array(info['Freq'], dtype=np.uint32)
    else:
      self.freq = None
    self.drawGraph()

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
    lis = []
    for i in range(self.list1.count()):    
      lis.append(str(self.list1.item(i).text()))
    info['List'] = lis
    if self.freq != None:
      info['Freq'] = self.freq.tolist()
    else:
      info['Freq'] = None  
    return info

  def measureLN2D(self):
    if self.parent.image_index >= 0:
      nsample = self.spin1.value() * 10000
      lnmax = self.spin2.value()
      seed = long(self.line1.text())
      self.freq = np.zeros((2, lnmax), dtype=np.uint32)
      self.list1.clear()
      self.button1.setEnabled(False) 
      self.button2.setEnabled(False)  
      self.button3.setEnabled(False)  
      self.button4.setEnabled(False)
      self.combo1.setEnabled(False)
      self.spin3.setEnabled(False)
      self.spin4.setEnabled(False)
      self.check1.setEnabled(False)
      self.check2.setEnabled(False)
      self.check3.setEnabled(False)
      self.pbar1.setValue(0)
      self.measure.setup(nsample, seed, self.parent.image_info, self.freq)        
      self.measure.start()

  def measureProgress(self, inc, filename):
    val = self.pbar1.value()
    self.pbar1.setValue(val+inc)
    self.list1.addItem(filename)

  def measureFinish(self):
    self.button1.setEnabled(True) 
    self.button2.setEnabled(True)     
    self.button3.setEnabled(True) 
    self.button4.setEnabled(True)
    self.combo1.setEnabled(True)
    self.spin3.setEnabled(True)
    self.spin4.setEnabled(True)
    self.check1.setEnabled(True)
    self.check2.setEnabled(True)
    self.check3.setEnabled(True)
    self.pbar1.setValue(100)
    self.spin4.setValue(int(self.measure.af*100.0))
    self.drawGraph()

  def drawGraph(self):
    if self.freq is None or self.insitu == False:
      return
    if self.combo1.currentIndex() == 0:
      if self.check3.isChecked():
        prob = self.ln2d_prob(0.01*self.spin4.value())
      else:
        prob = None    
      self.scene.drawLN2D(self.freq[0], 'LN2D', self.spin3.value(),\
        self.check1.isChecked(), self.check2.isChecked(), prob)
    elif self.combo1.currentIndex() == 1:
      if self.check3.isChecked():
        prob = self.ln2dr_prob(0.01*self.spin4.value())
      else:
        prob = None
      self.scene.drawLN2D(self.freq[1], 'LN2DR', self.spin3.value(),\
        self.check1.isChecked(), self.check2.isChecked(), prob)

  def poisson_prob(self, a, b):
    x = np.arange(b-1+0.1, 100, 0.1)
    y = np.zeros(x.size, dtype=np.float)
    for i in range(x.size):
      y[i] = m.pow(a, x[i]-b) / m.gamma(x[i]-b+1) * m.exp(-a)
    return [x, y, a+b, a]

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
    
  def showEvent(self, event):
    self.drawGraph()

  def saveCSV(self):    
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV Files (*.csv);;All Files (*.*)')
    if fname:
      fout = open(fname, 'w')  
      fout.write('Images,' + str(self.list1.count()) + '\n')
      for i in range(self.list1.count()):
        fout.write(self.list1.item(i).text() + '\n')
      fout.write('LNMax,' + str(len(self.freq[0])) + '\n')
      fout.write('Statistics, Total, Average, Variance\n')
      stat0 = self.scene.calcStat(self.freq[0])
      fout.write('GC' + ',' + str(stat0[0]) + ',' + str(stat0[1]) + ',' + str(stat0[2]) + '\n')
      stat1 = self.scene.calcStat(self.freq[1])
      fout.write('Random' + ',' + str(stat1[0]) + ',' + str(stat1[1]) + ',' + str(stat1[2]) + '\n')
      fout.write('LN, GC, GCRF, Random, RandomRF\n')
      rf0 = np.array(self.freq[0], dtype=np.float) / stat0[0]
      rf1 = np.array(self.freq[1], dtype=np.float) / stat1[0]
      for i in range(len(self.freq[0])):
        fout.write(str(i) + ',' + str(self.freq[0,i]) + ',' + str(rf0[i]) + ','\
          + str(self.freq[1,i]) + ',' + str(rf1[i]) + '\n')

  def saveGraph(self):
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save Graph', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')
    if fname:
      self.scene.saveGraph(str(fname))    

  def clearFreq(self):
    self.freq = None
    self.pbar1.setValue(0)
    self.list1.clear()
    self.scene.clearGraph()
    
"""
Size Dialog
"""
class SizeDialog(QtGui.QDialog):
  def __init__(self, parent):
    QtGui.QDialog.__init__(self, parent)
    self.setWindowTitle("Size")
    self.parent = parent
    self.insitu = True
    self.data = None
    hbox = QtGui.QHBoxLayout(self)
    vbox = QtGui.QVBoxLayout()
    hbox.addLayout(vbox)
    self.viewer = GraphicsView()
    self.viewer.setFixedSize(642, 482)
    self.scene = GraphScene()
    self.viewer.setScene(self.scene)
    hbox.addWidget(self.viewer)
    self.button1 = QtGui.QPushButton('Measure')
    self.button1.clicked[bool].connect(self.measureSize)
    vbox.addWidget(self.button1)
    self.list1 = QtGui.QListWidget()
    vbox.addWidget(self.list1)
    vbox.addWidget(QtGui.QLabel('Type :'))
    self.combo1 = QtGui.QComboBox()
    self.combo1.addItem('Diameter')
    self.combo1.addItem('Long side')    
    self.combo1.addItem('Narrow side')
    self.combo1.addItem('Aspect ratio')
    self.combo1.addItem('Angle')
    self.combo1.currentIndexChanged.connect(self.drawGraph)
    vbox.addWidget(self.combo1)
    vbox.addWidget(QtGui.QLabel('Bins :'))
    self.spin1 = QtGui.QSpinBox()
    self.spin1.setMinimum(1)
    self.spin1.setMaximum(500)
    self.spin1.setValue(30)
    self.spin1.valueChanged.connect(self.drawGraph)
    vbox.addWidget(self.spin1)
    self.check1 = QtGui.QCheckBox('Show Statistics')
    self.check1.setChecked(True)
    self.check1.stateChanged[int].connect(self.drawGraph)
    vbox.addWidget(self.check1)
    vbox.addStretch()    
    hbox1 = QtGui.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.button2 = QtGui.QPushButton('Save CSV')
    self.button2.clicked[bool].connect(self.saveCSV)
    hbox1.addWidget(self.button2)
    self.button3 = QtGui.QPushButton('Save Graph')
    self.button3.clicked[bool].connect(self.saveGraph)
    hbox1.addWidget(self.button3)
    self.button4 = QtGui.QPushButton('Close')
    self.button4.clicked[bool].connect(self.close)
    hbox1.addWidget(self.button4)

  def setInfo(self, info):
    self.insitu = False
    self.combo1.setCurrentIndex(info['Type'])
    self.spin1.setValue(info['Bins'])
    self.check1.setChecked(info['ShowStatistics'])
    self.insitu = True
    self.list1.clear()
    for it in info['List']:    
      self.list1.addItem(it)
    if info['Data'] != None:
      self.data = info['Data']
    else:
      self.data = None
    self.drawGraph()

  def getInfo(self):
    info = {}
    info['Type'] = self.combo1.currentIndex()
    info['Bins'] = self.spin1.value()
    info['ShowStatistics'] = self.check1.isChecked()
    lis = []
    for i in range(self.list1.count()):    
      lis.append(str(self.list1.item(i).text()))
    info['List'] = lis
    if self.data != None:
      info['Data'] = self.data
    else:
      info['Data'] = None  
    return info

  def measureSize(self):
    if self.parent.image_index < 0:
      return
    self.data = []
    self.list1.clear()
    for info in self.parent.image_info:
      fname = info['FileName']
      org_img = cv2.imread(fname, 1)
      roi_img = FileWidget.Process(org_img, info)
      filter_img = FilterWidget.Process(roi_img, info)
      thresh_img = ThresholdWidget.Process(filter_img, info)
      morphol_img = MorphologyWidget.Process(thresh_img, info)
      mod_img = ModifyWidget.ProcessMod(morphol_img, info)
      conts, hier = cv2.findContours(mod_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      filename = QtCore.QFileInfo(fname).fileName()
      self.list1.addItem(filename)
      for cont in conts:
        mom = cv2.moments(cont)
        if mom['m00'] != 0:
          x = int(mom['m10']/mom['m00'])
          y = int(mom['m01']/mom['m00'])
          area = cv2.contourArea(cont)
          dia = 2.0 * m.sqrt(area/m.pi)
          pos, size, ang = cv2.minAreaRect(cont)
          if size[0] >= size[1]:
            ls = size[0]
            ns = size[1]
          else:
            ls = size[1]
            ns = size[0]
            ang = ang + 90        
          self.data.append([str(filename), x, y, dia, ls, ns, ns/ls, ang])
    self.drawGraph()

  def drawGraph(self):
    if self.data == None or self.insitu == False:
      return
    ps = self.parent.pixelsize.data['PS']
    unit = self.parent.pixelsize.data['UNIT']
    if self.combo1.currentIndex() == 0:
      cdat = self.colData(3, ps)
      xlabel = 'Diameter (' + self.scene.unitText(unit) + ')'
    elif self.combo1.currentIndex() == 1:
      cdat = self.colData(4, ps)
      xlabel = 'Size of long side (' + self.scene.unitText(unit) + ')'
    elif self.combo1.currentIndex() == 2:
      cdat = self.colData(5, ps)
      xlabel = 'Size of narrow side (' + self.scene.unitText(unit) + ')'
    elif self.combo1.currentIndex() == 3:
      cdat = self.colData(6)
      xlabel = 'Aspect ratio'
    elif self.combo1.currentIndex() == 4:
      cdat = self.colData(7)
      xlabel = 'Angle (degree)'
    self.scene.drawSize(cdat, xlabel, self.spin1.value(), self.check1.isChecked())

  def showEvent(self, event):
    self.drawGraph()

  def colData(self, ind, ps=None):
    cdat = []
    if ps == None:
      for dat in self.data:
        cdat.append(dat[ind])
    else:
      for dat in self.data:
        cdat.append(dat[ind]*ps)        
    return np.array(cdat)

  def saveCSV(self):    
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save CSV', filter='CSV Files (*.csv);;All Files (*.*)')
    if fname:
      fout = open(fname, 'w')  
      fout.write('Images,' + str(self.list1.count()) + '\n')
      for i in range(self.list1.count()):
        fout.write(self.list1.item(i).text() + '\n')
      ps = self.parent.pixelsize.data['PS']
      unit = self.parent.pixelsize.data['UNIT']
      fout.write('PixelSize,' + str(ps) + ',' + unit + '\n')      
      fout.write('Statistics, NSample, Mean, STD\n')
      types = [[3, ps, 'Diameter'], [4, ps, 'LongSide'], [5, ps, 'NarrowSide'],\
        [6, None, 'AspectRatio'], [7, None, 'Angle']]
      for tp in types:
        cdat = self.colData(tp[0], tp[1])
        fout.write(tp[2] + ',' + str(cdat.shape[0]) + ',' + str(cdat.mean()) + ',' + str(cdat.std()) + '\n')
      fout.write('Filename, X, Y, Diameter, LongSide, NarrowSide, AspectRatio, Angle\n')
      for dat in self.data:
        fout.write(dat[0] + ',' + str(dat[1]*ps) + ',' + str(dat[2]*ps) + ','\
          + str(dat[3]*ps) + ',' + str(dat[4]*ps) + ',' + str(dat[5]*ps) + ','\
          + str(dat[6]) + ',' + str(dat[7]) + '\n')

  def saveGraph(self):
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save Graph', filter='Image Files (*.png *.pdf *.svg);;All Files (*.*)')
    if fname:
      self.scene.saveGraph(str(fname))
      
  def clearData(self):
    self.data = None
    self.list1.clear()
    self.scene.clearGraph()

"""
PixelSize Dialog
"""
class PixelSizeDialog(QtGui.QDialog):
  def __init__(self, parent, scene):
    QtGui.QDialog.__init__(self, parent)
    self.scene = scene
    self.data = {'PS': 1.0, 'UNIT': 'pixel'}
    self.scene.measurePixel[float].connect(self.measurePixel)
    self.insitu = True    
    self.setWindowTitle("Pixel Size")
    vbox = QtGui.QVBoxLayout(self)
    vbox.addWidget(QtGui.QLabel('Pixels :'))
    self.line1 = QtGui.QLineEdit()
    self.line1.setValidator(QtGui.QDoubleValidator())
    self.line1.textChanged[str].connect(self.calcPS)
    vbox.addWidget(self.line1)
    vbox.addWidget(QtGui.QLabel('Scale :'))
    hbox1 = QtGui.QHBoxLayout()
    vbox.addLayout(hbox1)
    self.line2 = QtGui.QLineEdit()
    self.line2.setValidator(QtGui.QDoubleValidator())
    self.line2.textChanged[str].connect(self.calcPS)
    hbox1.addWidget(self.line2)
    self.combo = QtGui.QComboBox()
    self.combo.addItem('pixel')    
    self.combo.addItem('km')    
    self.combo.addItem('m')      
    self.combo.addItem('cm')
    self.combo.addItem('mm')
    self.combo.addItem('um')    
    self.combo.addItem('nm')
    self.combo.currentIndexChanged[int].connect(self.calcPS)
    hbox1.addWidget(self.combo)
    vbox.addWidget(QtGui.QLabel('Pixel Size :'))
    self.label = QtGui.QLabel(str(self.data['PS']) + ' ' + self.data['UNIT'])
    vbox.addWidget(self.label)
    self.buttonb = QtGui.QDialogButtonBox()
    vbox.addWidget(self.buttonb)
    self.buttonb.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
    self.buttonb.accepted.connect(self.Accept)
    self.buttonb.rejected.connect(self.reject)    

  def measurePixel(self, dis):
    self.line1.setText(str(dis))

  def calcPS(self):
    if len(self.line1.text()) > 0 and len(self.line2.text()) > 0:
      pixels = float(self.line1.text())
      scale = float(self.line2.text())
      if pixels > 0:
        self.label.setText(str(scale/pixels) + ' ' + self.combo.currentText())

  def showEvent(self, event):
    self.line1.setText('')
    self.line2.setText('')
    self.label.setText(str(self.data['PS']) + ' ' + self.data['UNIT'])

  def Accept(self):
    ps = str(self.label.text()).split(' ')
    self.data['PS'] = float(ps[0])
    self.data['UNIT'] = ps[1]
    self.accept()

"""
MainWindow
"""    
class MainWindow(QtGui.QMainWindow):
  def __init__(self, parent=None):
    QtGui.QWidget.__init__(self, parent)
    self.setWindowTitle(self.tr("MPImage"))
    self.image_info = []
    self.image_index = -1
    self.current_file = None
    splitter = QtGui.QSplitter()
    self.setCentralWidget(splitter)
    self.MainMenuBar()
    self.MainToolBar()
    self.tab = QtGui.QTabWidget()
    self.tab.currentChanged[int].connect(self.SceneChanged)    
    #self.tab.setMinimumWidth(200)
    #self.tab.setMaximumWidth(400)
    self.viewer = GraphicsView()
    self.scene = ImageScene()
    self.viewer.setScene(self.scene)
    splitter.addWidget(self.tab)
    splitter.addWidget(self.viewer)
    self.file = FileWidget(self, self.scene)
    self.tab.addTab(self.file, 'Src')
    self.filter = FilterWidget(self, self.scene)
    self.tab.addTab(self.filter, 'Fil')
    self.threshold = ThresholdWidget(self, self.scene)
    self.tab.addTab(self.threshold, 'Thresh')
    self.morphology = MorphologyWidget(self, self.scene)    
    self.tab.addTab(self.morphology, 'Mor')
    self.modify = ModifyWidget(self, self.scene)
    self.tab.addTab(self.modify, 'Mod')
    self.pixelsize = PixelSizeDialog(self, self.scene)
    self.pixelsize.finished.connect(self.measurePSClose)
    self.imfp = IMFPDialog(self)
    self.ln2d = LN2DDialog(self)
    self.size = SizeDialog(self)

  def SceneChanged(self):
    self.scene.clearAll()
    self.scene.mouseMode = self.scene.mouseNone
    if self.tab.currentIndex() == 0:
      self.file.setImage()
    elif self.tab.currentIndex() == 1:
      self.filter.src_img = self.file.Process(self.file.org_img, self.file.getInfo())
      self.filter.setImage()
    elif self.tab.currentIndex() == 2:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      self.threshold.src_img = self.filter.Process(roi_img, self.filter.getInfo())
      self.threshold.setImage()
    elif self.tab.currentIndex() == 3:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      filter_img = self.filter.Process(roi_img, self.filter.getInfo())
      self.morphology.src_img = self.threshold.Process(filter_img, self.threshold.getInfo())
      self.morphology.setImage()
    elif self.tab.currentIndex() == 4:
      roi_img = self.file.Process(self.file.org_img, self.file.getInfo())
      filter_img = self.filter.Process(roi_img, self.filter.getInfo())
      thresh_img = self.threshold.Process(filter_img, self.threshold.getInfo())
      self.modify.src_img = self.morphology.Process(thresh_img, self.morphology.getInfo())
      if roi_img is not None:
        self.modify.org_img = roi_img.copy()
      self.modify.setImage()
      self.modify.setMouseMode()

  def MainMenuBar(self):
    menubar = QtGui.QMenuBar(self)
    self.setMenuBar(menubar)
    file_menu = QtGui.QMenu('File', self) 
    file_menu.addAction('Open', self.fileOpen)
    file_menu.addAction('Save', self.fileSave)
    file_menu.addAction('Save As', self.fileSaveAs)
    file_menu.addAction('Clear All', self.clearAllMsg)
    file_menu.addAction('Quit', QtCore.QCoreApplication.instance().quit)   
    menubar.addMenu(file_menu)
    mes_menu = QtGui.QMenu('Measure', self)
    mes_menu.addAction('Pixel Size', self.measurePS)
    mes_menu.addAction('IMFP', self.measureIMFP)
    mes_menu.addAction('LN2D', self.measureLN2D)
    mes_menu.addAction('Size', self.measureSize)
    menubar.addMenu(mes_menu)

  def clearAllMsg(self):
    msgbox = QtGui.QMessageBox(self)
    msgbox.setWindowTitle('Clear All')
    msgbox.setText('Do you want to clear all data ?')
    msgbox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No);
    msgbox.setDefaultButton(QtGui.QMessageBox.No);
    if msgbox.exec_() == QtGui.QMessageBox.Yes:
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
          msgbox = QtGui.QMessageBox(self)
          msgbox.setText("Can't find a image file, " + img_sname + '. Do you want to continue ?')
          msgbox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No);
          msgbox.setDefaultButton(QtGui.QMessageBox.No);
          if msgbox.exec_() == QtGui.QMessageBox.Yes:
            missing.append(image_info[i])
          else:
            return []
    if len(missing) > 0:
      for info in missing:
         image_info.remove(info)
    return image_info

  def fileOpen(self):
    fname = QtGui.QFileDialog.getOpenFileName(self, 'Open', filter='JSON Files (*.json);;All Files (*.*)')
    if fname:
      fin = open(fname, 'r')
      data = json.load(fin)      
      self.clearAll()
      self.image_info = self.checkImageInfo(fname, data['ImageInfo'])
      for info in self.image_info:
        self.image_list.addItem(QtCore.QFileInfo(info['FileName']).fileName())
      image_num = len(self.image_info)
      image_index = data['ImageIndex']
      if image_index < image_num:          
        self.image_list.setCurrentIndex(image_index)
      else:
        self.image_list.setCurrentIndex(image_num-1)
      self.scale_slider.setValue(data['ImageScale'])
      self.pixelsize.data = data['PixelSize']
      self.imfp.setInfo(data['IMFPInfo'])
      self.ln2d.setInfo(data['LN2DInfo'])
      self.size.setInfo(data['SizeInfo'])
      self.tab.setCurrentIndex(data['TabIndex'])
      self.current_file = fname 
      
  def fileSave(self):
    if self.current_file != None:
      fout = open(self.current_file, 'w')
      data = {}
      self.UpdateImageInfo()
      data['ImageInfo'] = self.image_info
      data['ImageIndex'] = self.image_list.currentIndex()
      data['ImageScale'] = self.scale_slider.value()
      data['PixelSize'] = self.pixelsize.data
      data['IMFPInfo'] = self.imfp.getInfo()
      data['LN2DInfo'] = self.ln2d.getInfo()
      data['SizeInfo'] = self.size.getInfo()
      data['TabIndex'] = self.tab.currentIndex()
      json.dump(data, fout)
      fout.close()
    else:
      self.fileSaveAs()

  def fileSaveAs(self):
    fname = QtGui.QFileDialog.getSaveFileName(self, 'Save', filter='JSON Files (*.json);;All Files (*.*)')
    if fname:
      self.current_file = fname
      self.fileSave()

  def measurePS(self):
    if self.pixelsize.isHidden():
      self.scene.mouseMode = self.scene.mouseMeasure
      self.pixelsize.show()
    else:
      self.pixelsize.activateWindow()

  def measurePSClose(self):
    self.scene.mouseMode = self.scene.mouseNone

  def measureIMFP(self):
    self.UpdateImageInfo()
    self.imfp.exec_()

  def measureLN2D(self):
    self.UpdateImageInfo()
    self.ln2d.exec_()

  def measureSize(self):
    self.UpdateImageInfo()
    self.size.exec_()

  def MainToolBar(self):
    toolbar = QtGui.QToolBar(self)
    self.addToolBar(toolbar)
    button1 = QtGui.QPushButton('Add Image')
    button1.clicked[bool].connect(self.AddImage)
    toolbar.addWidget(button1)
    button2 = QtGui.QPushButton('Delete Image')
    button2.clicked[bool].connect(self.DeleteImage)
    toolbar.addWidget(button2)    
    self.image_list = QtGui.QComboBox()    
    self.image_list.setMinimumWidth(120)
    self.image_list.currentIndexChanged[int].connect(self.ImageChanged)
    toolbar.addWidget(self.image_list)
    toolbar.addSeparator()
    button4 = QtGui.QPushButton('Save Image')
    button4.clicked[bool].connect(self.saveImage)
    toolbar.addWidget(button4)
    toolbar.addSeparator()
    self.scale_label = QtGui.QLabel('100%')
    self.scale_label.setFixedWidth(30)
    toolbar.addWidget(self.scale_label)        
    self.scale_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.scale_slider.setFixedWidth(200)
    self.scale_slider.setMinimum(5)
    self.scale_slider.setMaximum(200)
    self.scale_slider.setValue(100)
    self.scale_slider.valueChanged[int].connect(self.ChangeScale)
    toolbar.addWidget(self.scale_slider)
    button3 = QtGui.QPushButton('Fit')
    button3.setFixedWidth(35)
    button3.clicked[bool].connect(self.FitScale)
    toolbar.addWidget(button3)    

  def AddImage(self):
    fname = QtGui.QFileDialog.getOpenFileName(self, 'Add Image', filter='Image Files (*.jpg *.png *.tif *.bmp);;All Files (*.*)')
    if fname:
      info = {"FileName":str(fname)}
      info.update(self.file.getInfo(fname))
      info.update(self.filter.getInfo())
      info.update(self.threshold.getInfo())
      info.update(self.morphology.getInfo())
      info.update(self.modify.getInfo(False))
      self.image_info.append(info)
      self.image_list.addItem(QtCore.QFileInfo(fname).fileName())
      self.image_list.setCurrentIndex(self.image_list.count()-1)

  def DeleteImage(self):
    msgbox = QtGui.QMessageBox(self)
    msgbox.setText('Do you want to clear the current image ?')
    msgbox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No);
    msgbox.setDefaultButton(QtGui.QMessageBox.No);
    if msgbox.exec_() == QtGui.QMessageBox.Yes:
      self.image_index = -1
      index = self.image_list.currentIndex()
      self.image_info.pop(index)
      self.image_list.removeItem(index)

  def UpdateImageInfo(self):
    if self.image_index >= 0:   
      self.image_info[self.image_index].update(self.file.getInfo())
      self.image_info[self.image_index].update(self.filter.getInfo())
      self.image_info[self.image_index].update(self.threshold.getInfo())
      self.image_info[self.image_index].update(self.morphology.getInfo())
      self.image_info[self.image_index].update(self.modify.getInfo())   

  def ImageChanged(self):
    self.UpdateImageInfo()
    if self.image_list.currentIndex() >= 0:
      info = self.image_info[self.image_list.currentIndex()]
      self.file.setInfo(info)
      self.filter.setInfo(info)
      self.threshold.setInfo(info)
      self.morphology.setInfo(info)
      self.modify.setInfo(info)
      self.file.org_img = cv2.imread(info['FileName'], 1)
      self.SceneChanged()
      self.image_index = self.image_list.currentIndex()
    else:
      self.file.clearFile()
      self.morphology.clearLabel()
      self.scene.clearAll()

  def ChangeScale(self):
    val = self.scale_slider.value()
    self.scale_label.setText(str(val) + '%')
    self.scene.setScale(val / 100.0)
    
  def FitScale(self):
    s = int(self.scene.calcFitScale() * 100.0)
    self.scale_slider.setValue(s)

  def saveImage(self):
    if self.scene.pixmap != None:
      fname = QtGui.QFileDialog.getSaveFileName(self, 'Save image', filter='Image Files (*.jpg *.png *.bmp *.ppm);;All Files (*.*)')
      if fname:
        self.scene.pixmap.save(fname)
 
if __name__ == '__main__':
  app = QtGui.QApplication(sys.argv)
  window = MainWindow()
  window.show()
  sys.exit(app.exec_())


