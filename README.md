# SAM-Labelimg
利用Segment Anything(SAM)模型进行快速标注

#### 1.下载项目

项目1：https://github.com/zhouayi/SAM-Tool

项目2：https://github.com/facebookresearch/segment-anything

```bash
git clone https://github.com/zhouayi/SAM-Tool.git

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

下载`SAM`模型：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

#### 2.把数据放置在`<dataset_path>/images/*`这样的路径中，并创建空文件夹`<dataset_path>/embeddings`

#### 3.将项目1中的`helpers`文件夹复制到项目2的主目录下

##### 3.1 运行`extrac_embeddings.py`文件来提取图片的`embedding`

```bash
# cd到项目2的主目录下
python helpers\extract_embeddings.py --checkpoint-path sam_vit_h_4b8939.pth --dataset-folder <dataset_path> --device cpu
```

- `checkpoint-path`：上面下载好的`SAM`模型路径
- `dataset-folder`：数据路径
- `device`：默认`cuda`，没有`GPU`用`cpu`也行的，就是速度挺慢的

运行完毕后，`<dataset_path>/embeddings`下会生成相应的npy文件

##### 3.2 运行`generate_onnx.py`将`pth`文件转换为`onnx`模型文件

```bash
# cd到项目2的主目录下
python helpers\generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 1080 1920
```

- `checkpoint-path`：同样的`SAM`模型路径

- `onnx-model-path`：得到的`onnx`模型保存路径

- `orig-im-size`：数据中图片的尺寸大小`（height, width）`

【**注意：提供给的代码转换得到的`onnx`模型并不支持动态输入大小，所以如果你的数据集中图片尺寸不一，那么可选方案是以不同的`orig-im-size`参数导出不同的`onnx`模型供后续使用**】

#### 4.将生成的`sam_onnx.onnx`模型复制到项目1的主目录下，运行`segment_anything_annotator.py`进行标注

```bash
# cd到项目1的主目录下
python segment_anything_annotator.py --onnx-model-path sam_onnx.onnx --dataset-path <dataset_path> --categories cat,dog
```

- `onnx-model-path`：导出的`onnx`模型路径
- `dataset-path`：数据路径
- `categories`：数据集的类别（每个类别以`,`分割，不要有空格）

在对象位置出点击鼠标左键为增加掩码，点击右键为去掉该位置掩码。

#### 🆕 新增功能

**轮廓显示模式**：解决掩码遮挡图像内容的问题
- 按 `b` 键可在轮廓模式和填充模式之间切换
- 轮廓模式：仅显示边界线，不填充半透明区域，避免遮挡图像
- 填充模式：传统的半透明掩码填充

**标签显示控制**：解决文字标签遮挡的问题
- 按 `t` 键可显示/隐藏文字标签
- 隐藏标签后仍保留边界框，减少界面遮挡

其他使用快捷键有：

| `Esc`：退出app  | `a`：前一张图片 | `d`：下一张图片 |
| :-------------- | :-------------- | :-------------- |
| `k`：调低透明度 | `l`：调高透明度 | `n`：添加对象   |
| `r`：重置       | `Ctrl+s`：保存  | `Ctrl+z`：撤销  |
| **`b`：切换轮廓/填充模式** 🆕 | **`t`：显示/隐藏标签** 🆕 |                 |

![image](assets/catdog.gif)

最后生成的标注文件为`coco`格式，保存在`<dataset_path>/annotations.json`。

#### 5.检查标注结果
```bash
python cocoviewer.py -i <dataset_path> -a <dataset_path>\annotations.json
```
![image](assets/catdog.png)
#### 6.其他

- [ ] 修改标注框线条的宽度的代码位置

```python
# salt/displat_utils.py
class DisplayUtils:
    def __init__(self):
        self.transparency = 0.65 # 默认的掩码透明度
        self.box_width = 2 # 默认的边界框线条宽度
```

- [ ] 修改标注文本的格式的代码位置

```python
# salt/displat_utils.py
def draw_box_on_image(self, image, categories, ann, color):
    x, y, w, h = ann["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

    text = '{} {}'.format(ann["id"],categories[ann["category_id"]])
    txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 1.5, 1)[0]
    cv2.rectangle(image, (x, y + 1), (x + txt_size[0] + 1, y + int(1.5*txt_size[1])), color, -1)
    cv2.putText(image, text, (x, y + txt_size[1]), font, 1.5, txt_color, thickness=5)
    return image
```
```
- [ ] 2023.04.14 新增撤销上一个标注对象功能，快捷键Ctrl+z
- [x] 2025.11.21 新增轮廓显示模式，快捷键b，解决掩码遮挡问题
- [x] 2025.11.21 新增标签显示控制，快捷键t，解决文字遮挡问题
- [x] 2025.11.21 修复隐藏标注后点击重新显示的bug

## Reference
https://github.com/facebookresearch/segment-anything 

https://github.com/anuragxel/salt

https://github.com/trsvchn/coco-viewer
