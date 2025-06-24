from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from collections import OrderedDict

import os
import torch
from PIL import Image
import numpy as np
import models
from .convert_pidinet import convert_pidinet
from utils import *
from .pidinet import pidinet_converted

class EdgeDetector(torch.nn.Module):  # 修改：继承自 torch.nn.Module
    def __init__(self):
        """ 初始化模型 """
        super(EdgeDetector, self).__init__()  # 修改：调用父类构造函数
        self.args = self._get_args()  # 获取参数
        self.model = self._load_model()  # 加载模型

    def _get_args(self):
        """ 创建模拟的 args 对象（替代 argparse） """
        class Args:
            def __init__(self):
                self.model = 'pidinet_converted'  # 模型名称
                self.sa = True                   # 是否使用 CSAM
                self.dil = True                  # 是否使用 CDCM
                self.config = 'baseline'         # 配置名称
        return Args()

    def _load_model(self):
        """ 加载模型和权重 """
        from .pidinet import pidinet_converted  # 导入模型定义
        from .convert_pidinet import convert_pidinet  # 导入权重转换函数

        model = pidinet_converted(self.args)

        # 加载检查点
        checkpoint = self._load_checkpoint()
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除 "module." 前缀
            new_state_dict[name] = v

        # 转换并加载权重
        model.load_state_dict(convert_pidinet(new_state_dict, self.args.config))
        model.eval()  # 设置为评估模式
        return model

    def _load_checkpoint(self):
        """ 加载预训练权重 """
        model_filename = '/media/cb303/document/pose_game/deep-high-resolution-net.pytorch-master/lib/models/checkpoint_019.pth'
        print(f"=> loading checkpoint from '{model_filename}'")
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        return torch.load(model_filename, map_location='cpu')

    def forward(self, x):  # 修改：添加 forward 方法
        """
        前向传播方法，用于处理输入张量。
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            torch.Tensor: 处理后的张量
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        x = x.to(device)

        with torch.no_grad():
            results = self.model(x)
            result = torch.squeeze(results[-1])  # 提取最后一个输出
        return result

    def process_image(self, input_image_path, output_dir=None):
        """
        处理单张图像并返回结果

        Args:
            input_image_path (str): 输入图像路径
            output_dir (str, optional): 输出目录（可选，若提供则保存结果）

        Returns:
            PIL.Image.Image: 处理后的图像对象
        """
        # 加载图像
        image = Image.open(input_image_path).convert('RGB')
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 推理
        result_tensor = self.forward(image_tensor)  # 使用 forward 方法进行推理
        result = result_tensor.cpu().numpy()

        # 转换为图像对象
        result_img = Image.fromarray((result * 255).astype(np.uint8))

        # 保存结果（可选）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output.png")
            result_img.save(output_path)
            print(f"Processed and saved image: {output_path}")

        print(f"result_img shape after interpolation: {result_img.size}")
        return result_img
# 示例调用
if __name__ == "__main__":
    # 创建检测器实例
    detector = EdgeDetector()

    # 处理图像并获取结果
    input_path = "000000000036.jpg"
    output_dir = "put_one"
    result_image = detector.process_image(input_path, output_dir=output_dir)

    # 或者直接获取图像对象（不保存）
    # result_image = detector.process_image(input_path)