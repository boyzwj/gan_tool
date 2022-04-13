# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\project\pytorch_test\ui\FaceTool.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1350, 842)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1170, 740, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 851, 791))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.imgLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imgLabel.sizePolicy().hasHeightForWidth())
        self.imgLabel.setSizePolicy(sizePolicy)
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imgLabel.setObjectName("imgLabel")
        self.gridLayout.addWidget(self.imgLabel, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1160, 702, 71, 16))
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.previewCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.previewCheck.setEnabled(True)
        self.previewCheck.setGeometry(QtCore.QRect(870, 650, 71, 16))
        self.previewCheck.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.previewCheck.setTristate(False)
        self.previewCheck.setObjectName("previewCheck")
        self.refreshButton = QtWidgets.QPushButton(self.centralwidget)
        self.refreshButton.setGeometry(QtCore.QRect(880, 590, 75, 23))
        self.refreshButton.setObjectName("refreshButton")
        self.infoLabel = QtWidgets.QLabel(self.centralwidget)
        self.infoLabel.setGeometry(QtCore.QRect(880, 20, 191, 21))
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.infoLabel.setObjectName("infoLabel")
        self.resumeCheck = QtWidgets.QCheckBox(self.centralwidget)
        self.resumeCheck.setGeometry(QtCore.QRect(1050, 700, 61, 20))
        self.resumeCheck.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.resumeCheck.setObjectName("resumeCheck")
        self.sbBatchSize = QtWidgets.QSpinBox(self.centralwidget)
        self.sbBatchSize.setGeometry(QtCore.QRect(1240, 700, 42, 22))
        self.sbBatchSize.setMinimum(4)
        self.sbBatchSize.setMaximum(512)
        self.sbBatchSize.setSingleStep(4)
        self.sbBatchSize.setProperty("value", 32)
        self.sbBatchSize.setObjectName("sbBatchSize")
        self.sbPreview = QtWidgets.QSpinBox(self.centralwidget)
        self.sbPreview.setGeometry(QtCore.QRect(930, 690, 42, 22))
        self.sbPreview.setMinimum(4)
        self.sbPreview.setMaximum(64)
        self.sbPreview.setSingleStep(1)
        self.sbPreview.setProperty("value", 16)
        self.sbPreview.setObjectName("sbPreview")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(873, 690, 61, 21))
        self.label_2.setObjectName("label_2")
        self.sbImageSize = QtWidgets.QSpinBox(self.centralwidget)
        self.sbImageSize.setGeometry(QtCore.QRect(1240, 660, 42, 22))
        self.sbImageSize.setMinimum(64)
        self.sbImageSize.setMaximum(512)
        self.sbImageSize.setSingleStep(64)
        self.sbImageSize.setObjectName("sbImageSize")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1170, 660, 71, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1170, 620, 61, 21))
        self.label_4.setObjectName("label_4")
        self.sbLatentDim = QtWidgets.QSpinBox(self.centralwidget)
        self.sbLatentDim.setGeometry(QtCore.QRect(1240, 620, 42, 22))
        self.sbLatentDim.setMinimum(32)
        self.sbLatentDim.setMaximum(512)
        self.sbLatentDim.setSingleStep(1)
        self.sbLatentDim.setProperty("value", 256)
        self.sbLatentDim.setDisplayIntegerBase(10)
        self.sbLatentDim.setObjectName("sbLatentDim")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1350, 23))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionMaskface = QtWidgets.QAction(MainWindow)
        self.actionMaskface.setObjectName("actionMaskface")
        self.actionRectface = QtWidgets.QAction(MainWindow)
        self.actionRectface.setObjectName("actionRectface")
        self.actionThinface = QtWidgets.QAction(MainWindow)
        self.actionThinface.setObjectName("actionThinface")
        self.menuOpen.addAction(self.actionOpen)
        self.menuEdit.addAction(self.actionMaskface)
        self.menuEdit.addAction(self.actionRectface)
        self.menuEdit.addAction(self.actionThinface)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Train"))
        self.imgLabel.setText(_translate("MainWindow", "显示图片"))
        self.label.setText(_translate("MainWindow", "Batch Size"))
        self.previewCheck.setText(_translate("MainWindow", "实时预览"))
        self.refreshButton.setText(_translate("MainWindow", "刷新向量"))
        self.infoLabel.setText(_translate("MainWindow", "**********"))
        self.resumeCheck.setText(_translate("MainWindow", "Resume"))
        self.label_2.setText(_translate("MainWindow", "预览数量"))
        self.label_3.setText(_translate("MainWindow", "Image Size"))
        self.label_4.setText(_translate("MainWindow", "Latent Dim"))
        self.menuOpen.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionMaskface.setText(_translate("MainWindow", "Maskface"))
        self.actionRectface.setText(_translate("MainWindow", "Rectface"))
        self.actionThinface.setText(_translate("MainWindow", "Thinface"))
