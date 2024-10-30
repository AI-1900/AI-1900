import numpy as np

# 定义SiLU激活函数
class SiLU:
    def __init__(self):
        pass

    # 前向传播
    def forward(self, x):
        self.output = x / (1 + np.exp(-x))
        return self.output

    # 反向传播
    def backward(self, dout):
        # 计算SiLU函数的导数
        sig = 1 / (1 + np.exp(-self.output))
        dx = dout * (sig * (1 - sig))
        return dx

def test_main():
    # 创建SiLU实例
    silu = SiLU()

    # 测试数据
    x = np.array([1.0, -1.0, 2.0, -2.0])
    print("Input:", x)

    # 前向传播
    output = silu.forward(x)
    print("Forward output:", output)

    # 反向传播
    dout = np.array([1.0, 1.0, 1.0, 1.0])  # 假设上一层传递下来的梯度
    dx = silu.backward(dout)
    print("Backward output:", dx)

if __name__ == "__main__":
    test_main()