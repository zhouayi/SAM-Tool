import cv2
import numpy as np
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw, ImageFont

class DisplayUtils:
    def __init__(self):
        self.transparency = 0.3
        self.box_width = 2

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)
    
    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def overlay_mask_on_image(self, image, mask, color=(0, 0, 255)):
        gray_mask = mask.astype(np.uint8) * 255
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        color_mask = cv2.bitwise_and(gray_mask, color)
        masked_image = cv2.bitwise_and(image.copy(), color_mask)
        overlay_on_masked_image = cv2.addWeighted(
            masked_image, self.transparency, color_mask, 1 - self.transparency, 0
        )
        background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = ann["segmentation"]

        # 当边框刚好是4个值，可能会被认为矩形框，会出错，所以为4个值，再后面添加一个相同的点的坐标
        # TODO: 那万一出现两个点,再加一个点坐标就是4个值了，可能会出错，先放这里吧。
        # 解决错误地址：https://github.com/anuragxel/salt/issues/43
        for i in range(len(poly)):
            if len(poly[i]) == 4:
                poly[i] += poly[i][-2:]  # <---------------------------- Add same point again 

        rles = coco_mask.frPyObjects(poly, height, width)
        rle = coco_mask.merge(rles)
        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
        mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, categories, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

        text = '{} {}'.format(ann["id"],categories[ann["category_id"]])
        txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)

        # 增加中文的支持（原opencv不支持显示汉字）
        font = ImageFont.truetype("c:/windows/fonts/simhei.ttf", size=30)
        image = image[..., ::-1] 
        image = Image.fromarray(image, mode="RGB")
        draw = ImageDraw.Draw(image)
        txt_size = draw.textsize(text=text, font=font)
        # draw.text((x + 1, y + 1), text, fill=txt_color, font=font)
        draw.text((x + 1, y + 1), text, fill="red", font=font)
        image = np.asarray(image)
        image = image[..., ::-1]
        
        return image

    def draw_annotations(self, image, categories, annotations, colors):
        for ann, color in zip(annotations, colors):
            image = self.draw_box_on_image(image, categories, ann, color)
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            image = self.overlay_mask_on_image(image, mask, color)
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
