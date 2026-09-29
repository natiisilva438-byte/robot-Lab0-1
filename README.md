# 荣耀 200 手机摄像头内参标定：代码与原始数据集

本仓库仅公开 OpenCV 标定主程序和 **45 张未修改的原始 JPEG**。第一批有 31 张，补拍原图有 14 张；此前的压缩照片不在本数据集中。手机型号由拍摄者说明为荣耀 200，照片本身无法独立核实镜头和拍摄模式。

标定板为 **9 × 6 个方格**，即 **8 × 5 个内角点**，每格边长 **30 mm**。横拍原图为 4096 × 3072 像素；补拍数据中有 3 张以 3072 × 4096 竖拍保存，程序在检测时将其顺时针旋转 90°，不会改写原图。

## 内容

| 路径 | 说明 |
|---|---|
| `phone_calibration/calibrate.py` | 角点检测、方向统一、异常视图筛选及 OpenCV 内参标定主程序 |
| `phone_calibration/requirements.txt` | Python 依赖 |
| `phone_calibration/data/raw/` | 第一批 31 张原始 JPEG |
| `phone_calibration/data/supplemental_raw/` | 补拍的 14 张原始 JPEG |
| `phone_calibration/data/sha256.csv` | 每张原图的路径、SHA-256 和原始尺寸 |

## 运行

在仓库根目录执行：

```powershell
python -m pip install -r phone_calibration/requirements.txt
python phone_calibration/calibrate.py --input phone_calibration/data/raw --supplemental phone_calibration/data/supplemental_raw --output phone_calibration/results
```

程序将标定参数、逐图记录和角点预览写入本地 `phone_calibration/results/`。该目录已被 Git 忽略，不属于公开数据集。代码使用 `numpy.fromfile` 和 OpenCV `imdecode`，支持含中文的 Windows 路径。

## 许可

标定代码按 [MIT 许可证](LICENSE)发布；原始照片和数据清单按 [CC BY 4.0](DATA_LICENSE.md)发布。
