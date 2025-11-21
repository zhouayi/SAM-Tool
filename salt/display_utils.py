"""
显示工具模块

该模块提供了图像显示和标注可视化的工具类。
主要功能包括:
- 在图像上叠加掩码(mask)
- 绘制边界框和类别标签
- 绘制标注点
- 动态调整透明度和文字大小
- 支持轮廓模式和填充模式
- 支持标签显示控制
"""

import cv2
import numpy as np
from pycocotools import mask as coco_mask


class DisplayUtils:
    """
    显示工具类

    用于在图像上可视化各种标注信息,包括掩码、边界框、标签和点。
    支持动态调整显示参数如透明度和文字大小。

    属性:
        transparency (float): 掩码叠加的透明度,范围 [0.0, 1.0]
        box_width (int): 边界框线条宽度(像素)
        text_size (float): 文字大小,范围 [0.5, 5.0]
        contour_mode (bool): 是否使用轮廓模式(True=仅显示轮廓, False=填充掩码)
        show_labels (bool): 是否显示文字标签
        contour_thickness (int): 轮廓线条宽度(像素)
    """

    def __init__(self):
        """
        初始化显示工具

        设置默认的显示参数:
        - 透明度: 0.3
        - 边界框宽度: 2像素
        - 文字大小: 1.5
        - 轮廓模式: False (默认填充模式)
        - 显示标签: True
        - 轮廓宽度: 2像素
        """
        self.transparency = 0.3  # 掩码叠加透明度
        self.box_width = 2  # 边界框线条宽度
        self.text_size = 1.5  # 文字大小
        self.contour_mode = False  # 轮廓模式开关
        self.show_labels = True  # 标签显示开关
        self.contour_thickness = 2  # 轮廓线条宽度

    def increase_transparency(self):
        """
        增加透明度

        每次调用增加 0.05,最大值为 1.0(完全不透明)
        """
        self.transparency = min(1.0, self.transparency + 0.05)

    def decrease_transparency(self):
        """
        降低透明度

        每次调用减少 0.05,最小值为 0.0(完全透明)
        """
        self.transparency = max(0.0, self.transparency - 0.05)

    def increase_text_size(self):
        """
        增加文字大小

        每次调用增加 0.1,最大值为 5.0
        """
        self.text_size = min(5.0, self.text_size + 0.1)

    def decrease_text_size(self):
        """
        减小文字大小

        每次调用减少 0.1,最小值为 0.5
        """
        self.text_size = max(0.5, self.text_size - 0.1)

    def toggle_contour_mode(self):
        """
        切换轮廓模式

        在轮廓模式和填充模式之间切换:
        - 轮廓模式: 只绘制掩码的边界线,不填充内部
        - 填充模式: 以半透明颜色填充整个掩码区域
        """
        self.contour_mode = not self.contour_mode

    def toggle_labels(self):
        """
        切换标签显示

        在显示和隐藏文字标签之间切换。
        即使隐藏标签,边界框仍会显示。
        """
        self.show_labels = not self.show_labels

    def overlay_mask_on_image(self, image, mask, color=(0, 0, 255)):
        """
        在图像上叠加带颜色的掩码

        将二值掩码以指定颜色和透明度叠加到原始图像上。

        参数:
            image (numpy.ndarray): 原始图像,BGR 格式
            mask (numpy.ndarray): 二值掩码,形状为 (H, W),值为 0 或 1
            color (tuple): BGR 格式的颜色值,默认为红色 (0, 0, 255)

        返回:
            numpy.ndarray: 叠加掩码后的图像
        """
        # 将二值掩码转换为 0-255 的灰度图像
        gray_mask = mask.astype(np.uint8) * 255
        # 将单通道掩码转换为三通道
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        # 应用指定颜色到掩码
        color_mask = cv2.bitwise_and(gray_mask, color)
        # 从原始图像中提取掩码区域
        masked_image = cv2.bitwise_and(image.copy(), color_mask)
        # 将掩码区域与颜色按透明度混合
        overlay_on_masked_image = cv2.addWeighted(
            masked_image, self.transparency, color_mask, 1 - self.transparency, 0
        )
        # 提取背景区域(非掩码区域)
        background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
        # 将背景和叠加后的掩码合并
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def draw_mask_contour(self, image, mask, color=(0, 0, 255)):
        """
        在图像上绘制掩码轮廓线

        只绘制掩码的边界轮廓,不填充内部区域。
        适合在需要避免遮挡图像内容时使用。

        参数:
            image (numpy.ndarray): 原始图像,BGR 格式
            mask (numpy.ndarray): 二值掩码,形状为 (H, W),值为 0 或 1
            color (tuple): BGR 格式的轮廓颜色,默认为红色 (0, 0, 255)

        返回:
            numpy.ndarray: 绘制了轮廓的图像
        """
        # 将二值掩码转换为 uint8 格式
        mask_uint8 = (mask * 255).astype(np.uint8)
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制所有轮廓
        cv2.drawContours(image, contours, -1, color, self.contour_thickness)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        """
        将 COCO 格式的标注转换为掩码

        将多边形分割标注转换为二值掩码。

        参数:
            ann (dict): COCO 格式的标注,包含 'segmentation' 字段
            height (int): 图像高度
            width (int): 图像宽度

        返回:
            numpy.ndarray: 二值掩码,形状为 (height, width),dtype 为 uint8
        """
        # 初始化全零掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        # 获取多边形分割数据
        poly = ann["segmentation"]
        # 将多边形转换为 RLE 格式
        rles = coco_mask.frPyObjects(poly, height, width)
        # 合并多个 RLE
        rle = coco_mask.merge(rles)
        # 解码 RLE 为二值掩码
        mask_instance = coco_mask.decode(rle)
        # 反转掩码
        mask_instance = np.logical_not(mask_instance)
        # 将当前掩码与累积掩码合并
        mask = np.logical_or(mask, mask_instance)
        # 再次反转得到最终掩码
        mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, categories, ann, color):
        """
        在图像上绘制边界框和标签

        根据标注信息绘制边界框,并在框的上方显示对象 ID 和类别名称。
        标签的显示受 show_labels 属性控制。

        参数:
            image (numpy.ndarray): 要绘制的图像
            categories (dict): 类别 ID 到类别名称的映射字典
            ann (dict): 标注信息,包含 'bbox'(边界框)、'id'(对象 ID)和 'category_id'(类别 ID)
            color (tuple): BGR 格式的边界框颜色

        返回:
            numpy.ndarray: 绘制了边界框和标签的图像
        """
        # 提取边界框坐标(x, y, w, h 格式)
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 绘制矩形边界框
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

        # 如果启用标签显示,则绘制文字标签
        if self.show_labels:
            # 构建标签文本:对象 ID + 类别名称
            text = '{} {}'.format(ann["id"], categories[ann["category_id"]])
            # 根据背景颜色亮度选择文字颜色(深色背景用白字,浅色背景用黑字)
            txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 计算文字尺寸
            txt_size = cv2.getTextSize(text, font, self.text_size, 1)[0]
            # 根据文字大小动态计算线条粗细
            thickness = max(1, int(self.text_size * 3.33))

            # 绘制文字背景矩形
            cv2.rectangle(image, (x, y + 1), (x + txt_size[0] + 1, y + int(self.text_size * txt_size[1])), color, -1)
            # 在背景上绘制文字
            cv2.putText(image, text, (x, y + txt_size[1]), font, self.text_size, txt_color, thickness=thickness)
        return image

    def draw_annotations(self, image, categories, annotations, colors):
        """
        在图像上绘制所有标注

        遍历所有标注,为每个标注绘制边界框、标签和掩码。
        掩码的绘制方式由 contour_mode 属性控制:
        - contour_mode=True: 只绘制轮廓线
        - contour_mode=False: 填充半透明掩码

        参数:
            image (numpy.ndarray): 要绘制的图像
            categories (dict): 类别 ID 到类别名称的映射字典
            annotations (list): 标注列表,每个标注包含边界框、类别等信息
            colors (list): 颜色列表,与标注一一对应

        返回:
            numpy.ndarray: 绘制了所有标注的图像
        """
        # 遍历每个标注及其对应的颜色
        for ann, color in zip(annotations, colors):
            # 绘制边界框和标签
            image = self.draw_box_on_image(image, categories, ann, color)
            # 将标注转换为掩码
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            # 根据模式绘制掩码
            if self.contour_mode:
                # 轮廓模式:只绘制边界线
                image = self.draw_mask_contour(image, mask, color)
            else:
                # 填充模式:叠加半透明掩码
                image = self.overlay_mask_on_image(image, mask, color)
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        """
        在图像上绘制标注点

        根据标签使用不同颜色绘制点。常用于显示正负样本点。

        参数:
            image (numpy.ndarray): 要绘制的图像
            points (numpy.ndarray): 点坐标数组,形状为 (N, 2),每行为 (x, y)
            labels (numpy.ndarray): 点的标签数组,形状为 (N,)
            colors (dict): 标签到颜色的映射,默认 {1: 绿色(前景点), 0: 红色(背景点)}
            radius (int): 点的半径(像素),默认为 5

        返回:
            numpy.ndarray: 绘制了标注点的图像
        """
        # 遍历每个点
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            # 根据标签获取对应颜色
            color = colors[label]
            # 绘制实心圆点
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
