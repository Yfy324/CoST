import numpy as np


class Critic:

    def evaluate(self, inlier_scores, outlier_scores):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError


class Fpr(Critic):
    def __init__(self, recall_level=0.95):
        super().__init__()
        self.recall_level = recall_level

    def get_name(self):
        return 'FPR(' + str(self.recall_level * 100) + ')'

    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)  # 按行累加
        expected = np.sum(arr, dtype=np.float64)  # 所有元素求和
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                               'its last element does not correspond to sum')
        return out

    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                         np.array_equal(classes, [-1, 1]) or
                         np.array_equal(classes, [0]) or
                         np.array_equal(classes, [-1]) or
                         np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.   # 原pos为inlier/1，这里outlier样本为1 -- 需要修改为0.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]  # 返回由小到大排序后对应索引[::-1] --> 由大到小
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract  -- 对应第一行代码，提取不同的分数值对应索引 待定为阈值
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]  # diff(axis=-1): 后一列 - 前一列的值. where: 提取后小于前的索引
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]  # 保持维度，相当于把20000-1 加到distinct后面/加一行

        # accumulate the true positives with decreasing threshold： threshold是预测/变化的索引  tps是表示真正T/F变化的衡量
        tps = self.stable_cumsum(y_true)[threshold_idxs]  # 某个阈值下，正确识别inlier的个数
        fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing 某阈值下，错将out识别inlier的个数

        thresholds = y_score[threshold_idxs]  # 不同阈值

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1] 从15119索引开始倒序向前
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]  # 相同的softmax突变的地方 找fps、tps、对应的预测分

        cutoff = np.argmin(np.abs(recall - recall_level))  # 找到recall大于0.95对应的索引

        return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # fps[cutoff]/(fps[cutoff] + tps[cutoff]) 计算recall=0.95是对应的fps

    def evaluate(self, all_labels, all_scores):
        # all_scores = inlier_scores + outlier_scores
        # all_labels = [1 for _ in range(len(inlier_scores))] + [0 for _ in range(len(outlier_scores))]
        return self.fpr_and_fdr_at_recall(np.array(all_labels), np.array(all_scores), self.recall_level)