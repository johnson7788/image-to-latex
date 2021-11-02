import os
import subprocess
from pathlib import Path

import image_to_latex.data.utils as utils

# 下载数据集的位置
METADATA = {
    "im2latex_formulas.norm.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst",
    "im2latex_validate_filter.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst",
    "im2latex_train_filter.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst",
    "im2latex_test_filter.lst": "http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst",
    "formula_images.tar.gz": "http://lstm.seas.harvard.edu/latex/data/formula_images.tar.gz",
}
# 项目的目录地址: /media/wac/backup/john/johnson/image-to-latext
PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
DATA_DIRNAME = PROJECT_DIRNAME / "data"
RAW_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images"
PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"


def main():
    DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    cur_dir = os.getcwd()
    os.chdir(DATA_DIRNAME)

    # 下载图片和ground truth文件
    for filename, url in METADATA.items():
        if not Path(filename).is_file():
            utils.download_url(url, filename)

    # 解压
    if not RAW_IMAGES_DIRNAME.exists():
        RAW_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        utils.extract_tar_file("formula_images.tar.gz")

    # 提取感兴趣的部分, 搜索目录formula_images下的png图片，进行裁剪，保存到formula_images_processed目录下
    if not PROCESSED_IMAGES_DIRNAME.exists():
        PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("裁剪图片中...")
        for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
            cropped_image = utils.crop(image_filename, padding=8)
            if not cropped_image:
                continue
            cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

    # 清理ground truth 文件，替换im2latex_formulas.norm.lst中的一些字符，保存到im2latex_formulas.norm.new.lst
    cleaned_file = "im2latex_formulas.norm.new.lst"
    if not Path(cleaned_file).is_file():
        print("清理ground truth 数据...")
        script = Path(__file__).resolve().parent / "find_and_replace.sh"
        subprocess.call(["sh", f"{str(script)}", "im2latex_formulas.norm.lst", cleaned_file])

    # 创建单词表
    if not VOCAB_FILE.is_file():
        print("创建单词表...")
        # 读取公式文件中的所有公式，返回一个列表
        all_formulas = utils.get_all_formulas(cleaned_file)
        # 返回图片名字和公司
        _, train_formulas = utils.get_split(all_formulas, "im2latex_train_filter.lst")
        tokenizer = utils.Tokenizer()
        tokenizer.train(train_formulas)
        tokenizer.save(VOCAB_FILE)
    os.chdir(cur_dir)


if __name__ == "__main__":
    main()
