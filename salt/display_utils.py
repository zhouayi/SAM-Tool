import cv2
import numpy as np
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw, ImageFont

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont


class DisplayUtils:
    def __init__(self):
        self.transparency = 0.3
        self.box_width = 3
        self.pool_draw = ThreadPoolExecutor(max_workers=cpu_count() - 1)

        # 加载标签的字体
        try:
            try:
                # Should work for Linux
                self.font = ImageFont.truetype("simhei.ttf", size=20)
            except OSError:
                # Should work for Windows
                self.font = ImageFont.truetype("Arial.ttf", size=20)
        except OSError:
            # Load default, note no resize option
            # TODO: Implement notification message as popup window
            self.font = ImageFont.load_default()

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)
    
    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def _rle_to_mask(self, rle, height, width):
        # 这是 cocoviewer.py 中来的
        rows, cols = height, width
        rle_pairs = np.array(rle).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        index_offset = 0

        for index, length in rle_pairs:
            index_offset += index
            img[index_offset : index_offset + length] = 255
            index_offset += length

        img = img.reshape(cols, rows)
        img = img.T
        return img

    def overlay_mask_on_image(self, image, mask, color=(0, 0, 255)):
        gray_mask = mask.astype(np.uint8) * 255
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        color_mask = cv2.bitwise_and(gray_mask, color)
        masked_image = cv2.bitwise_and(image, color_mask)
        overlay_on_masked_image = cv2.addWeighted(
            masked_image, self.transparency, color_mask, 1 - self.transparency, 0
        )
        background = cv2.bitwise_and(image, cv2.bitwise_not(color_mask))
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        # polys结果是一个列表，代表着多个轮廓的坐标，
        # ploys[0]是一个一维列表，代表着第一个轮廓，like this [x1, y1, x2, y2, x3, y3, ...]
        # 做了兼容，原来的代码没取segmentation，后面自己改的代码取了传进来的
        polys = ann["segmentation"] if isinstance(ann, dict) else ann
 
        # 当边框刚好是4个值，可能会被认为矩形框，会出错，所以为4个值，再后面添加一个相同的点的坐标
        # TODO: 那万一出现两个点,再加一个点坐标就是4个值了，可能会出错，先放这里吧。
        # 解决错误地址：https://github.com/anuragxel/salt/issues/43
        for i in range(len(polys)):
            if len(polys[i]) == 4:
                polys[i] += polys[i][-2:]  # <---------------------------- Add same point again 

        """ 以下只是针对标注过程将mask显示出来而已，即使显示错误，对mask的标注结果是不影响的
        1、先创建一个纯黑的初始化mask;
        2、循环这些多边形的轮廓，每次都创建一个纯黑的temp_mask，然后用cv2.fillPoly把轮廓用白色填充;
        3、初始的mask与temp_mask，进行 XOR 逻辑亦或 运算，简单说就是(0+0=0, 0+1=1, 1+1=0),即两个条件，有且仅有一个为真时才为真;
        4、算第二个轮廓时，若是与第一个轮廓不相交，那就是0、1逻辑亦或为真，若是同心圆，那同心圆内部就是1+1=0，就把内部抠出来了;
        TODO：可能当三个同心圆，即奇数时，最中心最小那个同心圆，会因为是0+1=1，而扣不出来，但遇到概率不大，就是有，也问题不大。
        """
        if len(polys) == 1:
            # #（原来的方式）pycocotools中，针对同心圆，中间有空洞的这种mask画出来，中间的空心也会被填满，视觉上看上去不对
            mask = np.zeros((height, width), dtype=np.uint8)  # 原来的
            rles = coco_mask.frPyObjects(polys, height, width)
            rle = coco_mask.merge(rles)
            mask_instance = coco_mask.decode(rle)
            mask_instance = np.logical_not(mask_instance)
            mask = np.logical_or(mask, mask_instance)
            mask = np.logical_not(mask)
        else:
            mask = np.ones((height, width), dtype=np.uint8)
            for poly in polys:
                temp_mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(temp_mask, [pts], color=(1, 1, 1))
                mask = np.logical_xor(mask, temp_mask)   # 逻辑亦或
            mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, categories, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

        text = '{} {}'.format(ann["id"], categories[ann["category_id"]])
        txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)

        # 增加中文的支持（原opencv不支持显示汉字）
        # font = ImageFont.truetype("c:/windows/fonts/simhei.ttf", size=30)
        image = image[..., ::-1] 
        image = Image.fromarray(image, mode="RGB")
        draw = ImageDraw.Draw(image)
        txt_size = draw.textsize(text=text, font=self.font)
        # draw.text((x + 1, y + 1), text, fill=txt_color, font=self.font)
        draw.text((x + 1, y + 1), text, fill="red", font=self.font)
        image = np.asarray(image)
        image = image[..., ::-1]
        
        return image

    def draw_annotations(self, image, categories, annotations, colors, draw_mask=True):
        """
        # 1、这是原来的实现
        for ann, color in zip(annotations, colors):
            image = self.draw_box_on_image(image, categories, ann, color)
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            image = self.overlay_mask_on_image(image, mask, color)
        """

        """
        # 2、这是后面想通过多线程、不画mask的方式，提升不大。
        maskes = None
        if draw_mask:
            # # 先把所有的mask都算出来（性能提升大约是画20多个目标，从 2.56s 提升到 2.35s 这种，并不是很明显）
            # # 提交任务给线程池，并返回结果对象
            future_results = [self.pool_draw.submit(self.__convert_ann_to_mask, anno, image.shape[0], image.shape[1]) 
                            for anno in annotations]  
            # 等待所有任务完成，并获取结果
            maskes = [future_mask.result() for future_mask in future_results]
        
        if maskes is not None:
            for ann, mask, color in zip(annotations, maskes, colors):
                image = self.draw_box_on_image(image, categories, ann, color)
                image = self.overlay_mask_on_image(image, mask, color)
        else:
            # TODO:这里本来是只画框就好了，但只画框，循环两次后就会报错，所以就弄了一个假的掩码，仿着原来的才对，
            # 那后面有时间看排查一下，把画假掩码去掉。
            mask = np.zeros((image.shape[0], image.shape[1]))
            for ann, color in zip(annotations, colors):
                image = self.draw_box_on_image(image, categories, ann, color)
                image = self.overlay_mask_on_image(image, mask, color)
        
        """

        # 这是仿照 coco.view.py 写的，用PIL实现的，一张图28个目标，展示两张图，draw_annotations函数也调用了两次
        # 最原始的方法耗时 2.224s，下面这个耗时 0.354s, 快了大约6.3倍，以后再有类似的画图，来看看这
        # 传进来的格式是bgr的
        img_open = Image.fromarray(image[..., ::-1], mode="RGB").convert("RGBA")
        img_size = img_open.size
        # Create layer for bboxes and masks
        draw_layer = Image.new("RGBA", img_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(draw_layer)

        # 画mask
        masks = [anno["segmentation"] for anno in annotations]
        # 画框
        bboxes = [
            [
                anno["bbox"][0],
                anno["bbox"][1],
                anno["bbox"][0] + anno["bbox"][2],
                anno["bbox"][1] + anno["bbox"][3]
            ] 
            for anno in annotations
        ]

        # 画mask
        for i, (m, c) in enumerate(zip(masks, colors)):
            annotation = annotations[i]
            # TODO: 可把 128 设成 alpha 变量，然后通过界面点击来传参设置这个值
            # 这里 128 是设置mask的透明度的，范围是 [0, 255]， 0的话mask几乎看不到，255的话就几乎只看的到mask
            fill  = tuple(list(c) + [128])
            if isinstance(m, list):
                mask = self.__convert_ann_to_mask(m, image.shape[0], image.shape[1])
                mask = Image.fromarray(mask)
                draw.bitmap((0, 0), mask, fill=fill)
            # RLE mask for collection of objects (iscrowd=1)
            elif isinstance(m, dict) and annotation["iscrowd"]:
                mask = self._rle_to_mask(m["counts"][:-1], m["size"][0], m["size"][1])
                mask = Image.fromarray(mask)
                draw.bitmap((0, 0), mask, fill=fill)

        # 画框 (另起一个循环，不然先画的标签会被后画的mask盖住)
        for i, (b, c) in enumerate(zip(bboxes, colors)):
            annotation = annotations[i]
            draw.rectangle(b, outline=c, width=self.box_width)
            text = "{} {}".format(annotation["id"], categories[annotation["category_id"]])
            draw.text((b[0], b[1]), text, fill="red", font=self.font)

        image = Image.alpha_composite(img_open, draw_layer)
        image = np.asarray(image.convert("RGB"))
        # 这里一定要 .copy() 一下，不然 interface.py的60行中的QImage会报错
        # 解决办法来自：https://blog.csdn.net/weixin_44503976/article/details/130206803
        image = image[..., ::-1].copy()
        del draw
        
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            color = colors[label]
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
