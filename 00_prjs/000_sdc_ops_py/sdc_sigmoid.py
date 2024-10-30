import numpy as np

def sigmoid(x):
    """计算 Sigmoid 函数的值"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    """计算 Sigmoid 函数的导数"""
    return output * (1 - output)

def cross_entropy_loss(y_true, y_pred):
    """计算交叉熵损失"""
    # 为了避免对数的下溢，使用 np.clip 限制 y_pred 的范围
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def test_main():
    # 测试用例
    # 假设输入数据
    x = np.array([0.5, 1.0, 1.5])  # 模型的输入
    y_true = np.array([0.0, 1.0, 1.0])  # 真实标签

    # 前向传播
    y_pred = sigmoid(x)
    print("Sigmoid output (y_pred):", y_pred)

    # 计算损失
    loss = cross_entropy_loss(y_true, y_pred)
    print("Cross entropy loss:", loss)

    # 计算损失关于 Sigmoid 输出的梯度
    d_loss_d_y_pred = - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    # 反向传播
    d_loss_d_x = d_loss_d_y_pred * sigmoid_derivative(y_pred)
    print("Gradient of loss with respect to x:", d_loss_d_x)

if __name__ == "__main__":
    test_main()