"""
SAM-Tool 图形界面模块

本模块提供基于 PyQt5 的图形用户界面，用于图像标注和编辑。
包含自定义图形视图和应用程序主界面。
"""

import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel


class CustomGraphicsView(QGraphicsView):
    """
    自定义图形视图类

    继承自 QGraphicsView，提供图像显示、缩放和交互功能。
    支持鼠标滚轮缩放、鼠标点击标注等操作。
    """

    def __init__(self, editor):
        """
        初始化自定义图形视图

        参数:
            editor: 图像编辑器实例，用于处理标注逻辑
        """
        super(CustomGraphicsView, self).__init__()

        self.editor = editor

        # 设置渲染提示，提高显示质量
        self.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.setRenderHint(QPainter.SmoothPixmapTransform)  # 平滑图像变换
        self.setRenderHint(QPainter.TextAntialiasing)  # 文本抗锯齿

        # 设置优化标志，提升性能
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # 设置滚动条策略
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 水平滚动条按需显示
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 垂直滚动条按需显示

        # 设置变换和调整锚点为鼠标位置
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)

        # 创建并设置场景
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # 图像项初始化为空
        self.image_item = None

    def set_image(self, q_img):
        """
        设置要显示的图像

        参数:
            q_img: QImage 对象，要显示的图像
        """
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            # 如果已存在图像项，直接更新
            self.image_item.setPixmap(pixmap)
        else:
            # 否则创建新的图像项并添加到场景
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        """
        鼠标滚轮事件处理

        实现以鼠标位置为中心的图像缩放功能。
        向上滚动放大，向下滚动缩小。

        参数:
            event: 鼠标滚轮事件对象
        """
        zoom_in_factor = 1.25  # 放大系数
        zoom_out_factor = 1 / zoom_in_factor  # 缩小系数

        # 记录缩放前鼠标在场景中的位置
        old_pos = self.mapToScene(event.pos())

        # 根据滚轮方向确定缩放系数
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor  # 向上滚动，放大
        else:
            zoom_factor = zoom_out_factor  # 向下滚动，缩小

        # 执行缩放
        self.scale(zoom_factor, zoom_factor)

        # 计算缩放后鼠标位置的偏移并调整视图
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def imshow(self, img):
        """
        显示 OpenCV 格式的图像

        将 OpenCV 的 BGR 格式图像转换为 QImage 并显示。

        参数:
            img: OpenCV 格式的图像数组 (numpy.ndarray)
        """
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        # 转换为 QImage 并交换 R 和 B 通道（BGR -> RGB）
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        鼠标按键事件处理

        处理用户的标注点击操作：
        - 左键点击：添加正样本点（前景）
        - 右键点击：添加负样本点（背景）

        参数:
            event: 鼠标按键事件对象
        """
        pos = event.pos()
        # 将视图坐标转换为场景坐标
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()

        # 根据鼠标按键类型确定标签
        if event.button() == Qt.LeftButton:
            label = 1  # 左键：前景点
        elif event.button() == Qt.RightButton:
            label = 0  # 右键：背景点

        # 添加点击到编辑器并更新显示
        self.editor.add_click([int(x), int(y)], label)
        self.imshow(self.editor.display)


class ApplicationInterface(QWidget):
    """
    应用程序主界面类

    提供完整的图像标注界面，包括：
    - 顶部工具栏：操作按钮
    - 中央区域：图像显示和标注
    - 右侧面板：类别选择
    """

    def __init__(self, app, editor, panel_size=(1920, 1080)):
        """
        初始化应用程序界面

        参数:
            app: QApplication 实例
            editor: 图像编辑器实例
            panel_size: 面板尺寸，默认为 (1920, 1080)
        """
        super(ApplicationInterface, self).__init__()

        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        # 设置窗口标题，显示当前图像索引和总数
        self.setWindowTitle(f"1/{self.editor.dataset_explorer.get_num_images()}")

        # 创建主布局
        self.layout = QVBoxLayout()

        # 添加顶部工具栏
        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        # 创建主窗口水平布局
        self.main_window = QHBoxLayout()

        # 添加图形视图（中央显示区域）
        self.graphics_view = CustomGraphicsView(self.editor)
        self.main_window.addWidget(self.graphics_view)

        # 添加侧边面板（类别选择区域）
        self.panel = self.get_side_panel()
        self.main_window.addWidget(self.panel)
        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        # 显示初始图像
        self.graphics_view.imshow(self.editor.display)

    def reset(self):
        """
        重置当前标注

        清除当前图像上的所有未保存标注点和掩码。
        """
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def add(self):
        """
        添加标注对象

        保存当前标注的对象，并重置编辑器以开始新的标注。
        """
        self.editor.save_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def delet(self):
        """
        删除标注对象

        删除最后一个已保存的标注对象，并重置编辑器。
        """
        self.editor.delet_ann()
        self.editor.reset()
        self.graphics_view.imshow(self.editor.display)

    def next_image(self):
        """
        切换到下一张图像

        自动保存当前进度，每处理 10 张图像自动保存一次标注文件。
        """
        self.editor.next_image()
        self.graphics_view.imshow(self.editor.display)

        # 每处理 10 张图像自动保存一次
        if (self.editor.image_id + 1) % 10 == 0:
            self.editor.save()

        # 更新窗口标题显示当前进度
        self.setWindowTitle(f"{self.editor.image_id+1}/{self.editor.num_images}")

    def prev_image(self):
        """
        切换到上一张图像

        返回到前一张图像进行查看或修改。
        """
        self.editor.prev_image()
        self.graphics_view.imshow(self.editor.display)
        self.setWindowTitle(f"{self.editor.image_id+1}/{self.editor.num_images}")

    def toggle(self):
        """
        切换显示模式

        在原图和带标注信息的显示之间切换。
        """
        self.editor.toggle()
        self.graphics_view.imshow(self.editor.display)

    def transparency_up(self):
        """
        提高掩码透明度

        降低掩码的不透明度，使底层图像更清晰可见。
        """
        self.editor.step_up_transparency()
        self.graphics_view.imshow(self.editor.display)

    def transparency_down(self):
        """
        降低掩码透明度

        提高掩码的不透明度，使标注区域更明显。
        """
        self.editor.step_down_transparency()
        self.graphics_view.imshow(self.editor.display)

    def increase_text_size(self):
        """
        增大文字大小

        增大标注信息中文字标签的显示尺寸。
        """
        self.editor.increase_text_size()
        self.graphics_view.imshow(self.editor.display)

    def decrease_text_size(self):
        """
        减小文字大小

        减小标注信息中文字标签的显示尺寸。
        """
        self.editor.decrease_text_size()
        self.graphics_view.imshow(self.editor.display)

    def toggle_contour_mode(self):
        """
        切换轮廓模式

        在轮廓模式和填充模式之间切换。
        轮廓模式只显示边界线,避免遮挡图像内容。
        """
        self.editor.toggle_contour_mode()
        self.graphics_view.imshow(self.editor.display)

    def toggle_labels(self):
        """
        切换标签显示

        在显示和隐藏文字标签之间切换。
        隐藏标签可以减少界面遮挡。
        """
        self.editor.toggle_labels()
        self.graphics_view.imshow(self.editor.display)

    def save_all(self):
        """
        保存所有标注

        将当前所有标注数据保存到文件。
        """
        self.editor.save()

    def get_top_bar(self):
        """
        创建顶部工具栏

        包含所有常用操作按钮的工具栏。

        返回:
            QWidget: 包含所有操作按钮的顶部工具栏组件
        """
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)

        # 定义按钮列表：(按钮文本, 点击事件处理函数)
        buttons = [
            ("添加对象", lambda: self.add()),
            ("撤销对象", lambda: self.delet()),
            ("重置", lambda: self.reset()),
            ("上一张", lambda: self.prev_image()),
            ("下一张", lambda: self.next_image()),
            ("显示/隐藏标注", lambda: self.toggle()),
            ("轮廓/填充模式", lambda: self.toggle_contour_mode()),
            ("显示/隐藏标签", lambda: self.toggle_labels()),
            ("增加透明度", lambda: self.transparency_up()),
            ("降低透明度", lambda: self.transparency_down()),
            ("增大文字", lambda: self.increase_text_size()),
            ("减小文字", lambda: self.decrease_text_size()),
            ("保存标注", lambda: self.save_all()),
        ]

        # 创建并添加所有按钮
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            button_layout.addWidget(bt)

        return top_bar

    def get_side_panel(self):
        """
        创建侧边类别选择面板

        显示所有可用的标注类别，用户可通过单选按钮选择当前标注类别。

        返回:
            QWidget: 包含类别单选按钮的侧边面板组件
        """
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        categories = self.editor.get_categories()

        # 为每个类别创建单选按钮
        for category in categories:
            label = QRadioButton(category)
            # 绑定切换事件，传入选中的类别名称
            label.toggled.connect(lambda: self.editor.select_category(self.sender().text()))
            panel_layout.addWidget(label)

        return panel

    def keyPressEvent(self, event):
        """
        键盘按键事件处理

        提供快捷键操作支持：
        - Esc: 退出应用程序
        - A: 上一张图像
        - D: 下一张图像
        - K: 降低透明度
        - L: 增加透明度
        - B: 切换轮廓/填充模式
        - T: 切换标签显示/隐藏
        - N: 添加标注对象
        - R: 重置当前标注
        - Ctrl+S: 保存所有标注
        - Ctrl+Z: 撤销最后一个对象

        参数:
            event: 键盘按键事件对象
        """
        if event.key() == Qt.Key_Escape:
            self.app.quit()  # 退出程序
        if event.key() == Qt.Key_A:
            self.prev_image()  # 上一张
        if event.key() == Qt.Key_D:
            self.next_image()  # 下一张
        if event.key() == Qt.Key_K:
            self.transparency_down()  # 降低透明度
        if event.key() == Qt.Key_L:
            self.transparency_up()  # 增加透明度
        if event.key() == Qt.Key_B:
            self.toggle_contour_mode()  # 切换轮廓模式
        if event.key() == Qt.Key_T:
            self.toggle_labels()  # 切换标签显示
        if event.key() == Qt.Key_N:
            self.add()  # 添加对象
        if event.key() == Qt.Key_R:
            self.reset()  # 重置
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()  # 保存
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.delet()  # 撤销
