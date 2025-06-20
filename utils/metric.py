import numpy as np



def dice_coef(y_true, y_pred, num_classes):
    dice = np.zeros(num_classes)
    for label in range(num_classes):
        actual_positives = np.sum(y_true == label)  # 真实标签中类别 label 的数量
        predicted_positives = np.sum(y_pred == label)  # 预测标签中类别 label 的数量
        true_positives = np.sum((y_pred == label) & (y_true == label))  # 真正例

        # 如果真实标签中没有类别 label
        if actual_positives == 0:
            # 如果预测标签中也没有类别 label，则 Dice = 1
            if predicted_positives == 0:
                dice[label] = 1.0
            # 如果预测标签中有类别 label，则 Dice = 0
            else:
                dice[label] = 0.0
        else:
            # 正常计算 Dice
            denominator = predicted_positives + actual_positives
            dice[label] = (2 * true_positives) / denominator if denominator != 0 else 0
    return dice

def iou(y_true, y_pred, num_classes):
    ious = np.zeros(num_classes)
    for label in range(num_classes):
        actual_positives = np.sum(y_true == label)  # 真实标签中类别 label 的数量
        predicted_positives = np.sum(y_pred == label)  # 预测标签中类别 label 的数量
        intersection = np.sum((y_pred == label) & (y_true == label))  # 交集
        union = predicted_positives + actual_positives - intersection  # 并集

        # 如果真实标签中没有类别 label
        if actual_positives == 0:
            # 如果预测标签中也没有类别 label，则 IoU = 1
            if predicted_positives == 0:
                ious[label] = 1.0
            # 如果预测标签中有类别 label，则 IoU = 0
            else:
                ious[label] = 0.0
        else:
            # 正常计算 IoU
            ious[label] = intersection / union if union != 0 else 0
    return ious

def iou_with_dice(predict, labs,n_labels):
    predict = predict.cpu().detach().numpy()
    labs = labs.cpu().detach().numpy()
    predict = np.argmax(predict, axis=1)
    dice_pred = dice_coef(labs,predict, n_labels)
    iou_pred = iou(labs,predict,  n_labels)

    return dice_pred, iou_pred