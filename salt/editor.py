"""
SAM-Tool 编辑器模块

该模块提供图像标注编辑功能,集成了 SAM 模型进行交互式分割标注。
"""

import os, copy
import numpy as np

from salt.onnx_model import OnnxModel
from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils


class CurrentCapturedInputs:
    """当前捕获的输入数据管理类

    用于管理和存储用户交互过程中产生的点击输入、标签和掩码数据。
    这些数据会被传递给 SAM 模型进行图像分割预测。

    Attributes:
        input_point: 输入的点坐标数组,形状为 (N, 2),其中 N 为点的数量
        input_label: 输入点的标签数组,1 表示前景点,0 表示背景点
        low_res_logits: 低分辨率的 logits 输出,用于多次迭代预测
        curr_mask: 当前生成的分割掩码
    """

    def __init__(self):
        """初始化输入数据容器"""
        self.input_point = np.array([])  # 点击坐标数组
        self.input_label = np.array([])  # 点击标签数组(1:前景, 0:背景)
        self.low_res_logits = None  # 低分辨率预测结果,用于迭代优化
        self.curr_mask = None  # 当前生成的分割掩码

    def reset_inputs(self):
        """重置所有输入数据

        将所有输入点、标签、logits 和掩码清空,恢复到初始状态。
        通常在开始新的标注或切换图像时调用。
        """
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        """设置当前掩码

        Args:
            mask: 分割掩码数组,形状为 (H, W),数值为 0 或 1
        """
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        """添加用户点击输入

        将新的点击坐标和对应标签添加到输入序列中。支持多次点击累积。

        Args:
            input_point: 点击坐标,格式为 [x, y]
            input_label: 点击标签,1 表示前景点,0 表示背景点
        """
        if len(self.input_point) == 0:
            # 首次添加点,初始化数组
            self.input_point = np.array([input_point])
        else:
            # 追加新点到现有点序列
            self.input_point = np.vstack([self.input_point, np.array([input_point])])
        self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        """设置低分辨率 logits

        存储模型输出的低分辨率 logits,用于后续的迭代预测。
        这种机制可以提高多次点击时的分割精度。

        Args:
            low_res_logits: 模型输出的低分辨率 logits 数组
        """
        self.low_res_logits = low_res_logits


class Editor:
    """图像标注编辑器主类

    提供基于 SAM 模型的交互式图像分割标注功能。支持点击式标注、
    类别管理、标注保存等完整的数据集标注工作流。

    Attributes:
        dataset_path: 数据集根目录路径
        coco_json_path: COCO 格式的标注文件路径
        onnx_model_path: SAM ONNX 模型文件路径
        onnx_helper: ONNX 模型推理辅助类实例
        dataset_explorer: 数据集浏览和管理类实例
        curr_inputs: 当前输入数据管理实例
        categories: 类别列表
        image_id: 当前图像索引
        category_id: 当前选中的类别索引
        show_other_anns: 是否显示已有标注的标志
        num_images: 数据集中图像总数
        image: 当前图像数组 (RGB 格式)
        image_bgr: 当前图像数组 (BGR 格式,用于 OpenCV 显示)
        image_embedding: 当前图像的 SAM 特征嵌入
        display: 用于显示的图像副本
        du: 显示工具类实例
    """

    def __init__(self, onnx_model_path, dataset_path, categories=None, coco_json_path=None):
        """初始化编辑器

        Args:
            onnx_model_path: SAM ONNX 模型文件路径
            dataset_path: 数据集根目录路径
            categories: 类别名称列表,默认为 None(从 COCO 文件读取)
            coco_json_path: COCO 格式标注文件路径,默认为 None(使用默认路径)

        Raises:
            ValueError: 当 categories 和 coco_json_path 都未提供时抛出
        """
        self.dataset_path = dataset_path
        self.coco_json_path = coco_json_path
        self.onnx_model_path = onnx_model_path

        # 初始化 ONNX 模型推理器
        self.onnx_helper = OnnxModel(self.onnx_model_path)

        # 验证参数有效性
        if categories is None and not os.path.exists(coco_json_path):
            raise ValueError("categories must be provided if coco_json_path is None")

        # 设置默认标注文件路径
        if self.coco_json_path is None:
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")

        # 初始化数据集浏览器
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )

        # 初始化当前输入管理器
        self.curr_inputs = CurrentCapturedInputs()

        # 获取类别信息
        self.categories = self.dataset_explorer.get_categories()

        # 初始化状态变量
        self.image_id = 0  # 当前图像索引
        self.category_id = 0  # 当前类别索引
        self.show_other_anns = True  # 是否显示其他标注
        self.num_images = self.dataset_explorer.get_num_images()

        # 加载第一张图像及其嵌入
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)

        # 初始化显示图像
        self.display = self.image_bgr.copy()

        # 初始化显示工具
        self.du = DisplayUtils()

        # 重置编辑器状态
        self.reset()

    def add_click(self, new_pt, new_label):
        """添加点击输入并更新分割掩码

        接收用户点击的坐标和标签,调用 SAM 模型进行分割预测,
        并更新显示图像以展示分割结果。

        Args:
            new_pt: 点击坐标,格式为 [x, y]
            new_label: 点击标签,1 表示前景点,0 表示背景点
        """
        # 添加新的点击输入
        self.curr_inputs.add_input_click(new_pt, new_label)

        # 调用 SAM 模型进行分割预测
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )

        # 更新显示图像
        self.display = self.image_bgr.copy()
        # 只有在 show_other_anns 为 True 时才绘制已有标注
        if self.show_other_anns:
            self.draw_known_annotations()

        # 绘制点击点
        self.display = self.du.draw_points(
            self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
        )

        # 叠加预测掩码
        self.display = self.du.overlay_mask_on_image(self.display, masks[0, 0, :, :])

        # 保存当前掩码和 logits 用于后续迭代
        self.curr_inputs.set_mask(masks[0, 0, :, :])
        self.curr_inputs.set_low_res_logits(low_res_logits)

    def draw_known_annotations(self):
        """绘制已知标注

        在显示图像上绘制数据集中已存在的所有标注,
        包括边界轮廓和类别标签。
        """
        # 获取当前图像的标注和对应颜色
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        # 在图像上绘制标注
        self.display = self.du.draw_annotations(self.display, self.categories, anns, colors)

    def reset(self, hard=True):
        """重置编辑器状态

        清除当前的输入数据并刷新显示图像。

        Args:
            hard: 是否执行硬重置(当前未使用该参数,保留以便扩展)
        """
        # 清除所有输入数据
        self.curr_inputs.reset_inputs()

        # 重新生成显示图像
        self.display = self.image_bgr.copy()
        if self.show_other_anns:
            # 如果启用,则绘制已有标注
            self.draw_known_annotations()

    def toggle(self):
        """切换已有标注的显示状态

        在显示和隐藏其他标注之间切换,方便用户专注于当前标注。
        """
        self.show_other_anns = not self.show_other_anns
        self.reset()

    def step_up_transparency(self):
        """提高掩码透明度

        增加掩码叠加的透明度,使底层图像更清晰可见。
        """
        self.du.increase_transparency()
        self.reset()

    def step_down_transparency(self):
        """降低掩码透明度

        减少掩码叠加的透明度,使掩码更加明显。
        """
        self.du.decrease_transparency()
        self.reset()

    def increase_text_size(self):
        """增加文字大小

        增大标注文本的显示字体大小。
        """
        self.du.increase_text_size()
        self.reset()

    def decrease_text_size(self):
        """减小文字大小

        减小标注文本的显示字体大小。
        """
        self.du.decrease_text_size()
        self.reset()

    def toggle_contour_mode(self):
        """切换轮廓模式

        在轮廓模式和填充模式之间切换:
        - 轮廓模式: 只显示掩码边界线,不填充内部,避免遮挡图像
        - 填充模式: 使用半透明颜色填充掩码区域
        """
        self.du.toggle_contour_mode()
        self.reset()

    def toggle_labels(self):
        """切换标签显示

        在显示和隐藏文字标签之间切换。
        隐藏标签可以减少界面遮挡,但边界框仍会保留。
        """
        self.du.toggle_labels()
        self.reset()

    def save_ann(self):
        """保存当前标注

        将当前生成的分割掩码作为新标注添加到数据集中,
        关联当前选中的类别。
        """
        self.dataset_explorer.add_annotation(
            self.image_id, self.category_id, self.curr_inputs.curr_mask
        )

    def delet_ann(self):
        """删除当前图像的标注

        删除当前图像上的所有标注数据。

        注意:方法名拼写为 'delet' 而非 'delete',保持与原代码一致。
        """
        self.dataset_explorer.delet_annotation(self.image_id)

    def save(self):
        """保存标注到文件

        将所有标注数据以 COCO 格式保存到 JSON 文件中。
        """
        self.dataset_explorer.save_annotation()

    def next_image(self):
        """切换到下一张图像

        加载数据集中的下一张图像及其对应的嵌入特征,
        并重置编辑器状态。采用循环模式,最后一张后回到第一张。
        """
        # 更新图像索引(循环)
        self.image_id = (self.image_id + 1) % self.num_images

        # 加载新图像数据
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)

        # 更新显示并重置状态
        self.display = self.image_bgr.copy()
        self.reset()

    def prev_image(self):
        """切换到上一张图像

        加载数据集中的上一张图像及其对应的嵌入特征,
        并重置编辑器状态。采用循环模式,第一张前回到最后一张。
        """
        # 更新图像索引(循环)
        self.image_id = (self.image_id - 1) % self.num_images

        # 加载新图像数据
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)

        # 更新显示并重置状态
        self.display = self.image_bgr.copy()
        self.reset()

    def next_category(self):
        """切换到下一个类别

        将当前选中的类别索引移动到下一个,
        到达最后一个类别后循环回到第一个。
        """
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def prev_category(self):
        """切换到上一个类别

        将当前选中的类别索引移动到上一个,
        到达第一个类别后循环到最后一个。
        """
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1

    def get_categories(self):
        """获取类别列表

        Returns:
            list: 所有可用的类别名称列表
        """
        return self.categories

    def select_category(self, category_name):
        """根据类别名称选择类别

        通过类别名称设置当前选中的类别。

        Args:
            category_name: 要选择的类别名称

        Raises:
            ValueError: 当类别名称不存在时抛出
        """
        category_id = self.categories.index(category_name)
        self.category_id = category_id
