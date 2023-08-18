# SAM-Labelimg
利用Segment Anything(SAM)模型进行快速标注

## 1.下载项目

项目1：https://github.com/zhouayi/SAM-Tool

项目2：https://github.com/facebookresearch/segment-anything

```bash
git clone https://github.com/zhouayi/SAM-Tool.git

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

下载`SAM`模型：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## 2.把数据放置在`<dataset_path>/images/*`这样的路径中，并创建空文件夹`<dataset_path>/embeddings`

## 3.将项目1中的`helpers`文件夹复制到项目2的主目录下

### 3.1 运行`extrac_embeddings.py`文件来提取图片的`embedding`

```bash
# cd到项目2的主目录下
python helpers\extract_embeddings.py --checkpoint-path sam_vit_h_4b8939.pth --dataset-folder <dataset_path> --device cpu
```

- `checkpoint-path`：上面下载好的`SAM`模型路径
- `dataset-folder`：数据路径
- `device`：默认`cuda`，没有`GPU`用`cpu`也行的，就是速度挺慢的

运行完毕后，`<dataset_path>/embeddings`下会生成相应的npy文件

### 3.2 运行`generate_onnx.py`将`pth`文件转换为`onnx`模型文件

```bash
# cd到项目2的主目录下
python helpers\generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 1080 1920
```

- `checkpoint-path`：同样的`SAM`模型路径

- `onnx-model-path`：得到的`onnx`模型保存路径

- `orig-im-size`：数据中图片的尺寸大小`（height, width）`

【**注意：提供给的代码转换得到的`onnx`模型并不支持动态输入大小，所以如果你的数据集中图片尺寸不一，那么可选方案是以不同的`orig-im-size`参数导出不同的`onnx`模型供后续使用**】

## 4.将生成的`sam_onnx.onnx`模型复制到项目1的主目录下，运行`segment_anything_annotator.py`进行标注

```bash
# cd到项目1的主目录下
python segment_anything_annotator.py --onnx-model-path sam_onnx.onnx --dataset-path <dataset_path> --categories cat,dog
```

- `onnx-model-path`：导出的`onnx`模型路径
- `dataset-path`：数据路径
- `categories`：数据集的类别（每个类别以`,`分割，不要有空格）

在对象位置出点击鼠标左键为增加掩码，点击右键为去掉该位置掩码。

其他使用快捷键有：

| `Esc`：退出app  | `a`：前一张图片 | `d`：下一张图片 |
| :-------------- | :-------------- | :-------------- |
| `k`：调低透明度 | `l`：调高透明度 | `n`：添加对象   |
| `r`：重置       | `Ctrl+s`：保存  |                 |

![image](assets/catdog.gif)

最后生成的标注文件为`coco`格式，保存在`<dataset_path>/annotations.json`。

## 5.检查标注结果
```bash
python cocoviewer.py -i <dataset_path> -a <dataset_path>\annotations.json
```
![image](assets/catdog.png)
## 6.其他

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
- [ ] 2023.04.14新增加撤销上一个标注对象功能，快捷键Ctrl+z

---

## 7.说明

### 7.1 注意事项说明

1. 第一次打开加载数据有些慢，一定耐心等待，不要猛点。
2. 每一次启动后先去右边选中自己的目标类别。
3. 鼠标左键点击目标选中后，一些靠的很近的其它区域若是也被选中了，可以点击鼠标右键取消那些区域，或者直接点击“重置”，重新来；
   同样鼠标左键点击目标选中后，目标区域并未被完全选中的话，可再用鼠标左键点击目未被选中的区域。
4. 标错了(即出现了矩形框和标签后)，点击“撤销对象”或快捷键ctrl+z，可以取消。随后重新标注时标签上计数的数字会不会，但不影响。
5. 点击下一张时会自动保存，但退出之前还是记得保存一下先。
6. 鼠标滚轮可以缩小放大图片。
7. 第一次给的类别，后续就永远是这些类别的，只有先去把./dataset_path/annotations.json删掉。

### 7.2 新增特性及bug优化

1. 现已改为：每次打开软件时默认位置为上一次最后标注那张图所在位置。
2. 修复当前图像中没标注目标时点击“撤销对象”或快捷键ctrl+z时程序会退出的问题。
3. 已改为强制提醒，在未选择标注类别时会弹窗提醒。
4. 新增进度提示：点击窗口上方的 “显示当前进度” 可看到图像总数量及当前所在位置。
5. 新增界面打的标签支持中文，可参考[这](https://blog.csdn.net/qq_45945548/article/details/121316099)。（simhei.ttf在./assets文件夹中有）

- [ ] 在标注过快时，可能会意外退出。
- [x] TODO：bug，在界面上没有任何标注，上来直接点击“撤销对象”，程序依旧会退出。

---

## 8. 数据、环境准备及打包

注意：以及仅针对运行这个项目的环境，能够最小的打包。

### 8.1 环境及打包

以下环境准备好后就能运行此项目。（当然还需要准备好的数据）

```bash
conda create -n label python=3.8 -y
conda activate label

pip install  black==23.3.0  imageio==2.27.0 matplotlib==3.7.1 numpy==1.24.2 onnxruntime==1.14.1 opencv-python==4.7.0.72
pip install pillow==9.5.0 pycocotools==2.0.6 scikit-image==0.20.0  scipy==1.9.1 tomli==2.0.1
pip install PyQt5
pip install pyinstaller      # 为了打包
```

打包成exe：pyinstaller -F -w segment_anything_annotator.py

### 8.2 数据

按照上面的方法，得到的数据格式应如下：

./
│- └─dataset
│	    ├─embeddings
│    	│     00000.npy
│    	│     00001.npy
│    	│     00002.npy
│    	│	 .......
│    	└─images
│        	 00000.jpg
│         	00001.jpg
│         	00002.jpg
│    	 	.......

│- sam_onnx.onnx
│- segment_anything_annotator.exe



## Reference
https://github.com/facebookresearch/segment-anything 

https://github.com/anuragxel/salt

https://github.com/trsvchn/coco-viewer
