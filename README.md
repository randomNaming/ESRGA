# Enhanced Super-Resolution GAN Web Application

基于 Django 和 Real-ESRGAN 的图像超分辨率处理 Web 应用，支持 2x、4x、8x 放大倍数，可自动检测并使用 GPU/CPU 进行加速处理。

## ✨ 功能特性

- 🖼️ **图像超分辨率处理**：支持 2x、4x、8x 三种放大倍数
- 🚀 **自动硬件检测**：自动检测 GPU，有 GPU 时使用 GPU 加速，无 GPU 时自动切换到 CPU
- 🎯 **多种图像类型支持**：支持遥感图像、艺术图像、现实照片等多种类型
- 🌐 **Web 界面**：基于 Django 的友好 Web 界面，支持图片上传和下载
- 💾 **模型支持**：集成 Real-ESRGAN 预训练模型（x2plus、x4plus、x8）

## 📋 环境要求

- Python 3.8+
- Django 4.2.2
- PyTorch（CPU 或 CUDA 版本）
- 至少 4GB 内存（CPU 模式）或 4GB 显存（GPU 模式）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd Enhanced-Super-Resolution-Generative-Adversarial-Networks-Web-Based-On-Django
```

### 2. 安装依赖

#### 方式一：CPU 版本（适合 CPU 笔记本）

直接安装所有依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 方式二：GPU 版本（适合有 NVIDIA 显卡的电脑）

**步骤 1**：先安装 CUDA 版本的 PyTorch

```bash
# CUDA 11.8 版本（推荐，兼容性好）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**步骤 2**：安装其他依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证 GPU 是否可用**：

```bash
python -c "import torch; print('GPU可用:', torch.cuda.is_available()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

### 3. 下载模型权重文件

将模型权重文件放置在 `SR/weights/` 目录下：

- `RealESRGAN_x2plus.pth` - 2x 放大模型
- `RealESRGAN_x4plus.pth` - 4x 放大模型
- `RealESRGAN_x8.pth` - 8x 放大模型

> 💡 **提示**：模型文件下载链接：https://pan.quark.cn/s/1fdbd81c6682

### 4. 数据库迁移

进入 `SR` 目录并执行数据库迁移：

```bash
cd SR
python manage.py migrate
```

### 5. 启动服务器

```bash
python manage.py runserver
```

服务器启动后，在浏览器中访问：`http://127.0.0.1:8000`

## 📖 使用说明

1. **上传图片**：在 Web 界面中选择或拖放图片文件（建议 5MB 以内）
2. **选择放大倍数**：选择 2x、4x 或 8x 放大
3. **选择图像类型**：根据图片类型选择（遥感图像、艺术图像、现实图像）
4. **处理图片**：点击处理按钮，等待处理完成
5. **下载结果**：处理完成后可下载超分辨率后的图片

## 📁 项目结构

```
Enhanced-Super-Resolution-Generative-Adversarial-Networks-Web-Based-On-Django/
├── SR/                          # Django 项目主目录
│   ├── inference/               # 推理应用模块
│   ├── web_project/            # Django 项目配置
│   ├── templates/              # HTML 模板
│   ├── static/                 # 静态文件（CSS、JS、图片）
│   ├── weights/                 # 模型权重文件目录
│   ├── input/                   # 输入图片目录
│   ├── inferenceSR.py           # 核心推理代码（支持 CPU/GPU 自动检测）
│   └── manage.py               # Django 管理脚本
├── Real-ESRGAN/                # Real-ESRGAN 相关代码
├── requirements.txt            # Python 依赖列表
└── README.md                   # 项目说明文档
```

## ⚙️ 配置说明

### CPU/GPU 自动切换

项目已自动支持 CPU 和 GPU 模式切换：

- **GPU 模式**：自动检测到 GPU 时使用，速度更快，支持半精度加速
- **CPU 模式**：无 GPU 时自动切换，速度较慢但可以正常运行

代码会自动检测硬件环境，无需手动配置。

### 显存/内存优化

- **GPU 模式**：如果遇到显存不足（CUDA out of memory），可以在 `SR/inferenceSR.py` 中修改 `tile` 参数为 `400` 或 `512` 启用分块处理
- **CPU 模式**：已默认启用分块处理（`tile=400`），避免内存不足

## ⚠️ 注意事项

1. **CPU 模式速度较慢**：处理一张图片可能需要几分钟，请耐心等待
2. **图片大小限制**：建议输入图片分辨率不超过 1920x1080（CPU 模式）
3. **模型文件较大**：确保有足够的磁盘空间存储模型权重文件
4. **内存/显存要求**：
   - CPU 模式：建议至少 8GB 内存
   - GPU 模式：建议至少 4GB 显存（GTX 1650Ti 等）

## 🔧 常见问题

### Q: 安装 basicsr 时提示找不到 cv2 或 torch？

**A**: 请确保安装顺序正确，先安装 `opencv-python` 和 `torch`，再安装 `basicsr`。参考 `requirements.txt` 中的安装顺序。

### Q: 安装时提示 NumPy 版本不兼容？

**A**: 项目已锁定 `numpy==1.26.4`，如果遇到冲突，请先卸载旧版本：
```bash
pip uninstall numpy
pip install numpy==1.26.4
```

### Q: GPU 模式下显存不足？

**A**: 修改 `SR/inferenceSR.py` 中的 `tile` 参数，将 `tile=0` 改为 `tile=400` 或 `tile=512`。

### Q: 如何确认是否在使用 GPU？

**A**: 启动服务器后，处理图片时控制台会显示 "使用GPU加速: [GPU名称]" 或 "使用CPU模式"。

## 📚 相关资源

- **数据集下载**：https://pan.quark.cn/s/1fdbd81c6682
- **项目演示视频**：https://www.bilibili.com/video/BV1pM4y1J7Hq/?spm_id_from=333.999.0.0
- **Real-ESRGAN 官方仓库**：https://github.com/xinntao/Real-ESRGAN

## 👥 项目成员

- 伍彦来
- 何俊

## 📄 许可证

请查看 LICENSE 文件了解详情。

---

**提示**：如有问题，请查看控制台输出的错误信息，或参考常见问题部分。
