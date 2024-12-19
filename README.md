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

注：只是把图片抽取成numpy的数据，对python的版本，pytorch的版本要求不是特别大。

### 3.2 运行`generate_onnx.py`将`pth`文件转换为`onnx`模型文件

```bash
# cd到项目2的主目录下
python helpers\generate_onnx.py --checkpoint-path sam_vit_h_4b8939.pth --onnx-model-path ./sam_onnx.onnx --orig-im-size 1080 1920
```

- `checkpoint-path`：同样的`SAM`模型路径

- `onnx-model-path`：得到的`onnx`模型保存路径

- `orig-im-size`：数据中图片的尺寸大小`（height, width）`

【**注意：提供给的代码转换得到的`onnx`模型并不支持动态输入大小，所以如果你的数据集中图片尺寸不一，那么可选方案是以不同的`orig-im-size`参数导出不同的`onnx`模型供后续使用**】

注：为了导出成onnx：`pip install opencv-python pycocotools matplotlib onnxruntime onnx`  # 以下都是segment-anything项目的环境要求：

- segment-anything中要求的是 `python>=3.8`,`pytorch>=1.7`,`torchvision>=0.8`,只是上一步的抽取，版本要求不大，但是转onnx时版本不对就不行。
- 我用 python3.7 + torch1.8-cu111还是python3.8 + torch1.8-cu111 都是报错：
  “  File "/root/anaconda3/envs/s_sam/lib/python3.7/site-packages/torch/onnx/symbolic_helper.py", line 748, in _set_opset_version
      raise ValueError("Unsupported ONNX opset version: " + str(opset_version))
  ValueError: Unsupported ONNX opset version: 15”
- 最后是在docker镜像“nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04”中用的python3.9.16 + torch==2.0.1 （这个torch我是直接pip install的，它会自己安装cuda相关的库，或许它不需要带cuda的版本也能运行）（在备注详细些：onnx\==1.14.1、onnxruntime\==1.16.1）（注意一般直接Pip安装的torch的cuda版本都会超过比较老的显卡驱动的支持的最大cuda版本，就会用不了）

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

| `n`：添加对象             | `d`：下一张图片          | `a`：前一张图片 |
| :------------------------ | :----------------------- | :-------------- |
| `r`：重置(选中区域错误时) | `Ctrl+z`：撤销已标记目标 | `Ctrl+g`：跳转  |
| `k`：调低透明度           | `l`：调高透明度          | `Esc`：退出app  |

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
4. 标错了(即出现了矩形框和标签后)，点击“撤销对象”或快捷键ctrl+z，可以取消。随后重新标注时标签上计数的数字可能有错误，但不影响。
5. 点击上/下一张时会自动保存，但退出之前还是记得保存一下先（退出前也添加了自动保存）。
6. 鼠标滚轮可以缩小放大图片。

### 7.2 新增特性及bug优化

1. 现已改为：每次打开软件时默认位置为上一次最后标注那张图所在位置。
2. 修复当前图像中没标注目标时点击“撤销对象”或快捷键ctrl+z时程序会退出的问题。
3. 已改为强制提醒，在未选择标注类别时会弹窗提醒。
4. 新增进度提示：点击窗口上方的 “显示当前进度” 可看到图像总数量及当前所在位置。
5. 新增界面打的标签支持中文，可参考[这](https://blog.csdn.net/qq_45945548/article/details/121316099)。（simhei.ttf在./assets文件夹中有）
6. 3.2步骤中，由sam_vit_h_4b8939.pth 转成./sam_onnx.onnx，现在我已经将转换好的onnx传上来了，但大小应该是指定的 1024 1280 ，但好像用其它尺寸的图像也是可以做标注，可以先试一试，不行再去转换。
7. 新增**图片跳转**功能：点击窗口上方的“跳转”，输入跳转的序号，会跳到对应图片位置。
   绑定快捷键为 Ctrl+G
8. 新增异步数据保存，标注过程更加顺滑，且每存20次，会更新"backup.json"做数据备份，以防软件意外退出时，异步数据没写完而导致"annotations.json"错误。
9. 修复类似同心圆目标，有空洞存在时，无法显示空洞的bug。
10. 修复点击鼠标中间、鼠标侧键会错误退出的bug。
11. 修改画mask掩码从opencv的实现方式到pillow的实现方式，画面展示速度提升极大。
    - 在此基础上还增加了缓存策略，使得标注大量数据时也很丝滑。
12. 新增仅展示当前标注类别数据的功能。
13. 新增撤销对象也只能撤销当前选中的类别的对象，但同一类别的对象的撤销顺序暂目前只支持标注的倒序撤销。
14. 增加计算mask时的缓存策略，进一步提升软件速度，同时新增可选择点击区域时是否展示其它的标注mask。
15. 修复在空白图上，双击两次右键(即不选中任何目标)，点击添加会报错推出的bug。

- [x] ~~在标注过快时，可能会意外退出。报错"TypeError: Argument 'bb' has incorrect type (expected numpy.ndarray, got list)"~~
  已解决，bug原因[地址](https://github.com/anuragxel/salt/issues/43)。
- [x] ~~TODO：bug，在界面上没有任何标注，上来直接点击“撤销对象”，程序依旧会退出。~~已解决。

---

## 8. 数据、环境准备及打包

注意：以及仅针对运行这个项目的环境，能够最小的打包。

### 8.1 环境及打包

以下环境准备好后就能运行此项目。（当然还需要准备好的数据）

```bash
conda create -n label python=3.8 -y
conda activate label

pip install  black==23.3.0  imageio==2.27.0 matplotlib==3.7.1 numpy==1.24.2 onnxruntime==1.14.1 opencv-python==4.7.0.72  simplification==0.6.7  distinctipy==1.2.2
pip install pillow==9.5.0 pycocotools==2.0.6 scikit-image==0.20.0  scipy==1.9.1 tomli==2.0.1
pip install PyQt5
pip install pyinstaller      # 为了打包
```

打包成exe：pyinstaller -F -w segment_anything_annotator.py

### 8.2 数据

按照上面的方法，得到的数据格式应如下：

![image](assets/data_sample.png)

## Reference

https://github.com/facebookresearch/segment-anything 

https://github.com/anuragxel/salt

https://github.com/trsvchn/coco-viewer
