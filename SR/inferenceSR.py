from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import time
import torch
import numpy as np
from PIL import Image

def inferenceSR(filename,scale):
    # 自动检测是否有GPU可用
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        gpu_id = 0
        half = True  # GPU支持半精度，节省显存
        tile = 0  # GPU可以不分块
        print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
    else:
        gpu_id = None  # CPU模式
        half = False  # CPU不支持半精度
        tile = 400  # CPU模式建议使用分块，避免内存不足
        print("使用CPU模式（速度较慢，建议使用GPU）")
    
    model_8x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=8)
    model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_2x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    #netscale = 4
    print(scale)
    if scale == 4:
        upsampler = RealESRGANer(
                    scale=scale,
                    model_path='weights/RealESRGAN_x4plus.pth',
                    dni_weight=None,
                    model=model_4x,
                    tile=tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=half,
                    gpu_id=gpu_id)
    elif scale == 2:
        upsampler = RealESRGANer(
                    scale=scale,
                    model_path='weights/RealESRGAN_x2plus.pth',
                    dni_weight=None,
                    model=model_2x,
                    tile=tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=half,
                    gpu_id=gpu_id)
    elif scale == 8:
        upsampler = RealESRGANer(
                    scale=scale,
                    model_path='weights/RealESRGAN_x8.pth',
                    dni_weight=None,
                    model=model_8x,
                    tile=tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=half,
                    gpu_id=gpu_id)
    # 尝试使用OpenCV读取图像
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    # 如果OpenCV无法读取（比如多通道TIFF），使用PIL读取并转换
    if img is None or img.size == 0:
        try:
            # 使用PIL读取图像（PIL对多通道TIFF支持更好）
            pil_img = Image.open(filename)
            print(f"PIL读取图像成功，模式: {pil_img.mode}, 大小: {pil_img.size}")
            
            # 处理不同图像模式
            if pil_img.mode == 'RGB':
                # 已经是RGB，无需转换
                pass
            elif pil_img.mode == 'L':
                # 灰度图像，转换为RGB
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'RGBA':
                # 创建白色背景，合并alpha通道
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[3])
                pil_img = background
            elif pil_img.mode == 'LA':
                # 灰度+Alpha，转换为RGB
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'P':
                # 调色板模式，转换为RGB
                pil_img = pil_img.convert('RGB')
            else:
                # 处理多通道图像（如12通道TIFF）
                try:
                    # 尝试获取所有通道
                    channels = pil_img.split()
                    num_channels = len(channels)
                    print(f"检测到 {num_channels} 个通道，模式: {pil_img.mode}")
                    
                    if num_channels >= 3:
                        # 多通道图像，取前3个通道作为RGB
                        # 对于12通道图像，通常前3个通道是RGB或最相关的通道
                        r, g, b = channels[0], channels[1], channels[2]
                        pil_img = Image.merge('RGB', (r, g, b))
                        print(f"已从 {num_channels} 通道图像中提取前3个通道")
                    elif num_channels == 1:
                        # 灰度图像，转换为RGB
                        pil_img = pil_img.convert('RGB')
                    else:
                        # 2通道或其他情况，尝试转换为RGB
                        pil_img = pil_img.convert('RGB')
                except Exception as split_error:
                    # 如果split失败，尝试直接转换
                    print(f"通道分离失败: {split_error}，尝试直接转换")
                    pil_img = pil_img.convert('RGB')
            
            # 转换为numpy数组，然后转换为BGR（OpenCV格式）
            img_array = np.array(pil_img)
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.ndim == 2:
                img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"不支持的图像格式: shape={img_array.shape}, mode={pil_img.mode}")
            print(f"PIL转换成功，图像shape: {img.shape}")
        except Exception as e:
            raise ValueError(f"无法读取图像文件 {filename}: {str(e)}")
    
    # 检查图像是否成功读取
    if img is None or img.size == 0:
        raise ValueError(f"无法读取图像文件 {filename}，图像为空")
    
    # 确保图像是3通道BGR格式（Real-ESRGAN需要）
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        raise ValueError(f"图像通道数不正确: {img.shape[2]}，期望3通道（BGR）")
    
    cv2.imwrite('static/result/in.png', img)
    output, _ = upsampler.enhance(img, outscale=scale)
    cv2.imwrite('static/result/out.png', output)
    time.sleep(0.5)
    