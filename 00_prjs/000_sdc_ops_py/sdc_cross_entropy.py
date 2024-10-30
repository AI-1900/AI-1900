'''
Author: AI-1900 939614153@qq.com
Date: 2024-10-30 16:27:32
LastEditors: AI-1900 939614153@qq.com
LastEditTime: 2024-10-30 16:36:58
FilePath: \00_prjs\000_sdc_ops_py\sdc_cross_entropy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%A
'''
import numpy as np

def softmax(logits):
    """Compute softmax values for each sets of scores in logits."""
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """Compute the cross entropy loss."""
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

def cross_entropy_loss_derivative(y_true, y_pred):
    """Compute the derivative of the cross entropy loss."""
    return - (y_true / y_pred)

def cross_entropy_test():
    """Test the functions."""
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # Model output (logits)
    y_true = np.array([[0, 0, 1], [0, 1, 0]])  # True labels in one-hot encoding

    # Convert logits to probabilities using softmax
    probabilities = softmax(logits)

    # Compute the cross entropy loss
    loss = cross_entropy_loss(y_true, probabilities)
    print("Loss:", loss)

    # Compute the derivative of the loss for backpropagation
    d_loss = cross_entropy_loss_derivative(y_true, probabilities)
    print("Derivative:\n", d_loss)


if __name__ == "__main__":
    cross_entropy_test()