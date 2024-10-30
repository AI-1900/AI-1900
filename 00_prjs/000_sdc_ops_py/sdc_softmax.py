import numpy as np

def softmax(x):
    """Compute the softmax of the input vector x."""
    e_x = np.exp(x - np.max(x))  # subtract the max to prevent overflow
    return e_x / e_x.sum(axis=0)  # normalize the values

def test_main():
    # 测试用例
    # 假设 logits 是模型的原始输出，形状为 (batch_size, num_classes)
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # 两个样本，三个类别的 logits

    # 前向传播
    probabilities = softmax(logits)
    print("Softmax probabilities:\n", probabilities)

if __name__ == "__main__":
    test_main()