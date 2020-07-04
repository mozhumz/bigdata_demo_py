import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# 公式：根据预测概率排序之后
# (sum(正样本的index) - （正样本个数（正样本个数+1））/2) / 正负样本对个数
def auc_calculate(labels, preds, n_bins=100):
    postive_len = sum(labels)  # 正样本数量（因为正样本都是1）
    negative_len = len(labels) - postive_len  # 负样本数量
    total_case = postive_len * negative_len  # 正负样本对
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i] / bin_width)
        # print(nth_bin)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    # print("pos_histogram: ",pos_histogram)
    # print("neg_histogram: ",neg_histogram)

    # 累加负样本个数
    accumulated_neg = 0
    # 满足正负样本pair的个数
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]
    return satisfied_pair / float(total_case)


if __name__ == '__main__':
    y = np.array([1, 0, 0, 0, 1, 0, 1, 0, ])
    pred = np.array([0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7])

    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print("sklearn:", auc(fpr, tpr))
    print("验证:", auc_calculate(y, pred))

    # print(auc_calculate([1,0],[0.9,0.8]))
