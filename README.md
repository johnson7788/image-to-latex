# Image to LaTeX

图片转Latext公式
<img src="figures/screenshot.gif" alt="Image to Latex streamlit app" width="512">

## 简介

Deng等人（2016）]（https://arxiv.org/pdf/1609.04938v1.pdf）尝试了从图像到markup的生成问题。他们通过解析arXiv上的LaTeX论文来源，提取了大约100K的公式。他们使用pdflatex渲染公式，并将渲染后的PDF文件转换为PNG格式。他们的数据集的原始版本和预处理版本都可以得到[在线](http://lstm.seas.harvard.edu/latex/data/)。在他们的模型中，首先使用一个CNN来提取图像特征。然后用RNN对这些特征的行进行编码。最后，编码后的特征被一个带有注意力机制的RNN解码器使用。该模型总共有948万个参数。最近，Transformer在许多语言任务中已经超越了RNN，所以我想我可以在这个问题上试一试。

## 方法

使用他们的数据集，我训练了一个模型，使用ResNet-18作为编码器，采用二维位置编码，使用Transformer作为解码器，采用交叉熵损失。
(与[Singh等人(2021)](https://arxiv.org/pdf/2103.06450.pdf)中描述的相似，只是我只使用了ResNet到第3块以减少计算成本，而且我排除了数字编码，因为它不适用于这个问题）。该模型有大约300万个参数。

<img src="figures/model_architecture.png" alt="Model Architecture" width="384">

<small>Model architecture. Taken from Singh et al. (2021).</small>

最初，我使用预处理的数据集来训练我的模型，因为为了提高效率，预处理的图像被降到原始尺寸的一半，
并被分组和填充成类似的尺寸，以方便批次。然而，这种僵化的预处理被证明是一个巨大的限制。
尽管该模型在测试集（与训练集的预处理方式相同）上可以达到合理的性能，但它对数据集以外的图像没有很好的概括性，
很可能是因为图像质量、填充和字体大小与数据集中的图像有很大不同。
其他使用相同数据集尝试相同问题的人也观察到了这种现象（例如，[本项目](https://wandb.ai/site/articles/image-to-latex)、[this issue](https://github.com/harvardnlp/im2markup/issues/12)和[this issue](https://github.com/harvardnlp/im2markup/issues/21)）。

为此，我使用了原始数据集，并在我的数据处理pipeline中加入了图像增强（例如随机缩放、高斯噪声），
以增加样本的多样性。此外，与Deng等人（2016）不同的是，我没有按大小对图像进行分组。
相反，我对它们进行了统一采样，并将它们填充到批次中最大的图像的大小，这样，模型必须学会如何适应不同的填充大小。

我在数据集中遇到的其他问题。
- 一些latex代码产生了视觉上相同的输出（例如：`left(`和`right)`看起来和`(`和`)`一样），所以我把它们归一化了。
- 一些latex代码被用来添加空格（例如``vspace{2px}`和`hspace{0.3mm}`）。
  然而，空间的长度即使对人类来说也是难以判断的。而且，有很多方法来表达相同的间距（例如，1厘米=10毫米）。
  最后，我不希望模型在空白图像上生成代码，所以我删除了它们。
  我只删除了`vspace`和`hspace`，但事实证明有很多命令用于水平间距。我在错误分析中才意识到这一点。见下文。

## Results

[best run]](https://wandb.ai/kingyiusuen/image-to-latex/runs/1w1abmg1/)在测试集的字符错误率(CER)为0.17。下面是测试数据集中的一个例子。

<img width="480" src="https://user-images.githubusercontent.com/14181114/131140417-38d2e647-8316-41d5-9b81-583ecd2668a0.png">

- 输入的图像和模型的预测看起来是一样的。但是在ground truth标签中，水平间隔是用`~`创建的，而模型使用的是`\`，所以这仍然被算作一个错误。

我还在一些随机的维基百科文章中拍了一些截图，看看这个模型是否能推广到数据集之外的图像。

<img width="480" alt="Screen Shot 2021-08-27 at 8 06 54 AM" src="https://user-images.githubusercontent.com/14181114/131131947-fd857bd6-17e9-4a00-87d0-6ba04442c730.png">

- 模型的输出实际上是正确的，但是由于某些原因，Streamlit不能用`\cal`渲染代码。

<img width="480" src="https://user-images.githubusercontent.com/14181114/131130008-867e7373-67cb-44fb-abdb-b2eb2b6d6dd9.png">

- 错误地加粗了一些符号。

当图像大于数据集中的那些图像时，模型似乎也有一些问题。也许我应该在数据增强的过程中增加重新缩放系数的范围。

## 讨论

我想我应该更好地界定项目的范围。

- 我想让模型区分普通大小的括号和大括号（例如：`(`, `\big(`, `\Big(`, `\bigg(`, `\Bigg(`)）？
- 我想让模型识别水平和垂直间距吗？(有[40多个关于水平间距的命令](https://tex.stackexchange.com/a/74354)。)
- 我想让模型识别不同的字体风格吗？(这里有[LaTex中可用的字体样式列表](https://tex.stackexchange.com/a/58124)。)
- 等等。

这些问题应被用来指导数据清理过程。

我发现了一个相当成熟的工具，叫做[Mathpix Snip](https://mathpix.com/)，
可以将手写的公式转换为LaTex代码。它的[vocabulary size](https://docs.mathpix.com/#vocabulary)约为200。
不包括数字和英文字母，它能产生的LaTex命令的数量实际上刚刚超过100。(im2latex-100k的单词表量差不多是500）。
它只包括两个水平间距命令（`quad'和`qquad'），而且它不能识别不同大小的括号。
Perphas限制在一组有限的单词表中是我应该做的，因为现实世界中的LaTeX有太多的模糊之处。

这项工作明显可能的改进包括：（1）对模型进行更多的epoch训练（为了节省时间，我只训练了15个epoch，但验证损失仍在下降），
（2）使用beam search（我只实现了贪婪搜索），
（3）使用更大的模型（例如，使用ResNet-34而不是ResNet-18）并做一些超参数调整。
我没有做这些，因为我的计算资源有限（我使用的是谷歌Colab）。但最终，我相信拥有没有模糊标签的数据和做更多的数据增量是这个问题成功的关键。

模型的表现并不尽如人意，但我希望我从这个项目中学到的经验对将来想解决类似问题的人有用。

## 怎样使用

### 设置

克隆版本库到你的电脑上，把你的命令行放在版本库文件夹内。

```
git clone https://github.com/kingyiusuen/image-to-latex.git
cd image-to-latex
```
然后，创建一个名为 "venv "的虚拟环境并安装所需的软件包。

```
make venv
make install-dev
```

### 数据预处理

运行以下命令，下载im2latex-100k数据集并进行所有的预处理。(图像裁剪步可能需要一个多小时。)

```
python scripts/prepare_data.py
```

### 模型训练和实验跟踪

#### 模型训练

训练命令

```
python scripts/run_experiment.py trainer.gpus=1 data.batch_size=32
```

配置可以在`conf/config.yaml`或命令行中修改。参见[Hydra的文档](https://hydra.cc/docs/intro)以了解更多。
训练完成模型保存在，是一个ckpt文件:
```angular2html
outputs/2021-11-02/12-00-04/wandb/run-20211102_120006-1ksnqcgn/files/image-to-latex/1ksnqcgn/checkpoints/epoch=13-val/loss=0.12-val/cer=0.06.ckpt
```

#### 使用权重和偏差进行实验跟踪

最好的模型checkpoint将被自动上传到Weights & Biases (W&B)（在训练开始前，你将被要求注册或登录W&B）。下面是一个从W&B下载训练过的模型checkpoint的命令样本。

```
python scripts/download_checkpoint.py RUN_PATH
```
用你的运行路径替换RUN_PATH。运行路径的格式应该是`<entity>/<project>/<run_id>`。要找到一个特定实验运行的运行路径，请到仪表板的Overview选项卡。

例如，你可以使用以下命令来下载我的最佳运行状态的checkpoint

```
python scripts/download_checkpoint.py kingyiusuen/image-to-latex/1w1abmg1
```
checkpoint将被下载到项目目录下一个名为`artifacts`的文件夹。

### 测试和持续集成

以下工具被用来对代码库进行润色。

`isort`: 对Python脚本中的导入语句进行排序和格式化。

`black`: 一个遵守PEP8的代码格式化器。

`flake8`: 一个报告Python脚本stylistic问题的代码linter。

`mypy`: 在Python脚本中执行静态类型检查。

使用以下命令来运行所有的检查器和格式化器。

```
make lint
```

它们的配置见根目录下的`pyproject.toml`和`setup.cfg`。

当提交时，预提交框架会自动进行类似的检查。请查看`.pre-commit-config.yaml`的配置。

### 部署

创建了一个API来使用训练好的模型进行预测。使用下面的命令来启动和运行服务器。

```
make api
```

你可以通过生成的文档（http://0.0.0.0:8000/docs）探索该API。
要运行Streamlit应用程序，创建一个新的终端窗口并使用以下命令。
```
make streamlit
```
该应用程序应在你的浏览器中自动打开。你也可以通过访问[http://localhost:8501]（http://localhost:8501）来打开它。
为了使该应用程序工作，你需要下载一个实验运行的artifacts（见上文），并使API启动和运行。

要为API创建一个Docker镜像。

```
make docker
```

## 训练数据
```bash
.
├── formula_images    #原始的公式图像
├── formula_images.tar.gz    #压缩包，解压后得到formula_images
├── formula_images_processed  # 经过处理后的公式图像，去掉没用的像素，
├── im2latex_formulas.norm.lst #原始的图像对应的latex的公式TXT文件
├── im2latex_formulas.norm.new.lst  #经过scripts下的find_and_replace.sh替换后
├── im2latex_test_filter.lst  #格式是 图像名称 对应的im2latex_formulas.norm.new.lst中的latex的公式的行数，
├── im2latex_train_filter.lst
└── im2latex_validate_filter.lst  # 例如5abbb9b19f.png 0，代表这里面的formula_images_processed/5abbb9b19f.png 图像对应的latext是im2latex_formulas.norm.new.lst 第0行
```

## Acknowledgement

- This project is inspired by the project ideas section in the [final project guidelines](https://docs.google.com/document/d/1pXPJ79cQeyDk3WdlYipA6YbbcoUhIVURqan_INdjjG4/edit) of the course [Full Stack Deep Learning](https://fullstackdeeplearning.com/) at UC Berkely. Some of the code is adopted from its [labs](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/tree/main).

- [MLOps - Made with ML](https://madewithml.com/courses/mlops/) for introducing Makefile, pre-commit, Github Actions and Python packaging.

- [harvardnlp/im2markup](https://github.com/harvardnlp/im2markup) for the im2latex-100k dataset.
