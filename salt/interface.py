from PyQt5 import QtGui
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent, QCloseEvent
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox, QInputDialog

from .editor import Editor


class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        old_pos = self.mapToScene(event.pos())
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()
        if event.button() == Qt.LeftButton:
            label = 1
        elif event.button() == Qt.RightButton:
            label = 0
        self.editor.add_click([int(x), int(y)], label)
        self.imshow(self.editor.display)


class ApplicationInterface(QWidget):
    def __init__(self, app, editor: Editor, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()

        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        self.main_window = QHBoxLayout()

        self.graphics_view = CustomGraphicsView(self.editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.main_window.addWidget(self.panel)
        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.graphics_view.imshow(self.editor.display)

    def reset(self):
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        self.editor.save_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def delet(self):
        self.editor.delet_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def next_image(self):
        if self.editor.image_id + 1 == self.editor.imgs_num:
            QMessageBox.warning(self, "警告", "这已经是最后一张图了！")
            return

        self.editor.next_image()
        self.graphics_view.imshow(self.editor.display)
        self.editor.save()

    def prev_image(self):
        # 注意这种实例化弹窗方式与上面直接调用静态方法，静态方法更简便，
        # 但下面实例化对象后，可以其它时候再执行，且.exec_()会有返回值，代表用户点的那个按钮。
        if self.editor.image_id == 0:
            msg_box = QMessageBox(QMessageBox.Warning, "警告", "已经是第一张图了！")
            msg_box.exec_()
            return

        self.editor.prev_image()
        self.graphics_view.imshow(self.editor.display)
        self.editor.save()

    def toggle(self):
        self.editor.toggle()
        self.graphics_view.imshow(self.editor.display)

    def process(self):
        # 为了显示一下进度
        info = "当前进度：{}/{}".format(self.editor.image_id + 1, self.editor.imgs_num)
        msg_box = QMessageBox(QMessageBox.Information, "进度", info)
        msg_box.exec_()

    def jump(self):
        # 为了图片跳转
        value, success = QInputDialog.getInt(
            self, "图片跳转", f"请输入[1,{self.editor.imgs_num}]中的整数: ", value=self.editor.image_id + 1, min=1, max=self.editor.imgs_num, step=1)
        if success:
            self.editor.image_id = value - 1 - 1  # 为了显示，给的值都比索引大1，所以减去1
            self.editor.next_image()   # 然后这里面对self.editor.image_id加了1，所以上面还要再减去1，
            self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        self.editor.step_up_transparency()
        self.graphics_view.imshow(self.editor.display)

    def transparency_down(self):
        self.editor.step_down_transparency()
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        self.editor.save()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        buttons = [
            ("添加对象", lambda: self.add()),
            ("撤销对象", lambda: self.delet()),
            ("重置", lambda: self.reset()),
            ("前一张", lambda: self.prev_image()),
            ("下一张", lambda: self.next_image()),
            ("显示已标注信息", lambda: self.toggle()),
            ("显示当前进度", lambda: self.process()),
            ("跳转", lambda: self.jump()),
            ("调高透明度", lambda: self.transparency_up()),
            ("调低透明度", lambda: self.transparency_down()),
            ("保存", lambda: self.save_all()),
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories = self.editor.get_categories()
        for category in categories:
            # label = QPushButton(category)
            # label.clicked.connect(lambda: self.editor.select_category(category))
            # panel_layout.addWidget(label)

            label = QRadioButton(category)
            # sender传入点击的字符
            label.toggled.connect(
                lambda: self.editor.select_category(self.sender().text()))
            panel_layout.addWidget(label)
        return panel

    # 这个函数是继承于父类的，这里我重写了,采用异步保存数据，必须等线程池先写完，不然直接退出json文件就坏了。
    # 这是点击右上角的x关闭按钮触发的事件
    def closeEvent(self, event: QCloseEvent) -> None:
        reply = QMessageBox.question(
            self, u'警告', u'确认退出?', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()  # 关闭窗口
            # 点击确定后，再保存一下
            self.save_all()

            # 之后得留时间让线程池的保存执行完，不然会损坏json文件
            self.editor.pool.close()
            self.editor.pool.join()
        else:
            event.ignore()  # 忽视点击X事件
        # return super().closeEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Escape:
            # 退出前再保存一下
            self.save_all()
            # 在关闭软件前先关闭线程池，并通过join阻塞祖先成，文件保存完后才退出。
            self.editor.pool.close()
            self.editor.pool.join()
            self.app.quit()
        
        # Qt.Key.Key_Left与Qt.Key_Left一个意思，后者没加枚举类的类名而已
        # 在这里， Qt.Key.Key_Left 代表 <- 按键不起作用，可能是被过滤或者其他上层窗口捕捉了 
        if event.key() == Qt.Key_A:
            self.prev_image()
        if event.key() == Qt.Key_D:
            self.next_image()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.delet()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key.Key_G:
            self.jump()
        # elif event.key() == Qt.Key_Space:
        #     # Do something if the space bar is pressed
        #     pass
