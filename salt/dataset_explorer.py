from pycocotools import mask
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy
from datetime import datetime


def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "year": int(datetime.now().strftime("%Y")),
            "contributor": "SH",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append(
            {"id": i, "name": category, "supercategory": category}
        )
    for i, image_name in enumerate(image_names):
        im = cv2.imread(os.path.join(dataset_folder, image_name))
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id, poly=False):
    start_anno_id = anno_id
    x, y, width, height = bounding_box_from_mask(image_mask)
    if poly == False:
        fortran_binary_mask = np.asfortranarray(image_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
    if poly == True:
        contours = measure.find_contours(image_mask, 0.5)
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [],
    }
    if poly == False:
        annotation["segmentation"] = encoded_mask
        annotation["segmentation"]["counts"] = str(
            annotation["segmentation"]["counts"], "utf-8"
        )
    if poly == True:
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            sc = bunch_coords(segmentation)
            sc = simplify_coords_vwp(sc, 2)
            sc = unbunch_coords(sc)
            annotation["segmentation"].append(sc)
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, coco_json_path=None):
        self.dataset_folder = dataset_folder
        self.image_names = os.listdir(os.path.join(self.dataset_folder, "images"))
        self.image_names.sort()  # 因为os.listdir的特性一定要排序一下，不然标注结果很可能不是按预期从第一张图到最后一张图
        self.image_names = [
            os.path.split(name)[1] for name in self.image_names if name.endswith(".jpg") or name.endswith(".png")
        ]
        self.image_names.sort()
        self.coco_json_path = coco_json_path
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)
        with open(coco_json_path, "r", encoding="utf-8") as f:
            self.coco_json = json.load(f)

        self.categories = [category["name"] for category in self.coco_json["categories"]]
        self.annotations_by_image_id = {}
        for annotation in self.coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)

        self.global_annotation_id = len(self.coco_json["annotations"])
        self.category_colors = distinctipy.get_colors(len(self.categories))
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]

        # 为了得到最后标注的图片的id
        self.last_img_id = 0
        if self.coco_json["annotations"]:
            self.last_img_id = self.coco_json["annotations"][-1].get("image_id")
        
        # 得到所有图片的数量
        self.imgs_num = len(self.coco_json["images"])

        # 因为使用了异步存数据，为了确保安全，每保存20次，就会再写一份json当做备份
        self.async_nums = 0
        self.backup_json_path = os.path.join(os.path.dirname(self.coco_json_path), "backup.json")

        # 创建一个变量来记录标注是否变化，变化了才会存json文件，而不是每次下一张都存，会极大改善快速连续下一张的卡顿感
        self.save_flag = False

    def __init_coco_json(self, categories):
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]
    
    def get_categories(self):
        return self.categories

    def get_num_images(self):
        return len(self.image_names)

    def get_image_data(self, image_id):
        image_name = self.coco_json["images"][image_id]["file_name"]
        image_path = os.path.join(self.dataset_folder, image_name)
        embedding_path = os.path.join(
            self.dataset_folder,
            "embeddings",
            os.path.splitext(os.path.split(image_name)[1])[0] + ".npy",
        )
        image = cv2.imread(image_path)
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_embedding = np.load(embedding_path)
        return image, image_bgr, image_embedding

    def __add_to_our_annotation_dict(self, annotation):
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)
    
    def __delet_to_our_annotation_dict(self, image_id):
        return self.annotations_by_image_id[image_id].pop(-1)

    def get_annotations(self, image_id, return_colors=False):
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def add_annotation(self, image_id, category_id, mask, poly=True):
        if mask is None:
            return
        annotation = parse_mask_to_coco(
            image_id, self.global_annotation_id, mask, category_id, poly=poly
        )
        self.__add_to_our_annotation_dict(annotation)
        self.coco_json["annotations"].append(annotation)
        self.global_annotation_id += 1
        self.save_flag = True

    def delet_annotation(self, image_id):
        # 加个判断，避免多点然后直接退出
        # if self.annotations_by_image_id[image_id]: 
        if self.annotations_by_image_id.get(image_id, []):  # 避免直接开始就点撤销的空字典的“KeyError”错误
            # 确保  self.coco_json 中删除的数据和图片显示是一致的，不能直接用 pop(-1)
            # 之前的方式是无法修改之前的标错的数据的，界面显示正确，但是coco删的是错的。
            # 现在这种方式就一个缺点，删除之前的，coco中的注释的id不是连续完整的，但完全不影响
            annotation = self.__delet_to_our_annotation_dict(image_id)
            index = self.coco_json["annotations"].index(annotation)
            self.coco_json["annotations"].pop(index)
            self.global_annotation_id -= 1
            self.save_flag = True

    def save_annotation(self):
        # 仅当有标注增加、减少时，调用此函数时才会去保存
        if not self.save_flag:
            return

        with open(self.coco_json_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False是为了保存json时，中文就是中文，不会自动转为unicode字符
            json.dump(self.coco_json, f, ensure_ascii=False)
    
        # 这里是加的异步的备份数据保存：
        if self.async_nums % 20 == 0:
            with open(self.backup_json_path, "w", encoding="utf-8") as f:
                json.dump(self.coco_json, f, ensure_ascii=False)
            
        self.async_nums += 1
        self.save_flag = False

    def get_last_anno_img_id(self):
        # 为了得到最后标注的一张的图片的id
        return self.last_img_id
    
    def get_imgs_num(self):
        # 得到所有的图片数量
        return self.imgs_num
