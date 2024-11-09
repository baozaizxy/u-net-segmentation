import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT):
    SR_classes = torch.argmax(SR, dim=1)
    correct_predictions = (SR_classes == GT).sum()
    total_pixels = GT.numel()
    accuracy = float(correct_predictions) / total_pixels
    return accuracy

def get_accuracy(SR_pred, GT):
    correct_predictions = (SR_pred == GT).sum()
    total_pixels = GT.numel()
    accuracy = float(correct_predictions) / total_pixels
    return accuracy

def get_sensitivity(SR, GT, class_label=1):
    SR_class = torch.argmax(SR, dim=1)
    TP = ((SR_class == class_label) & (GT == class_label)).sum().item()
    FN = ((SR_class != class_label) & (GT == class_label)).sum().item()
    se = TP / (TP + FN + 1e-6)
    return se

def get_specificity(SR, GT, num_classes=4, background_label=0):
    SR_classes = torch.argmax(SR, dim=1)
    specificities = []
    for class_label in range(num_classes):
        if class_label == background_label:
            continue
        SR_class = (SR_classes == class_label).float()
        GT_class = (GT == class_label).float()
        TN = ((SR_class == 0) & (GT_class == 0)).sum()
        FP = ((SR_class == 1) & (GT_class == 0)).sum()
        SP = float(TN) / (float(TN + FP) + 1e-6)
        specificities.append(SP)
    return sum(specificities) / len(specificities) if specificities else 0.0

def get_precision(SR, GT, num_classes=4, background_label=0):
    SR_classes = torch.argmax(SR, dim=1)
    precisions = []
    for class_label in range(num_classes):
        if class_label == background_label:
            continue
        SR_class = (SR_classes == class_label).float()
        GT_class = (GT == class_label).float()
        TP = ((SR_class == 1) & (GT_class == 1)).sum()
        FP = ((SR_class == 1) & (GT_class == 0)).sum()
        PC = float(TP) / (float(TP + FP) + 1e-6)
        precisions.append(PC)
    return sum(precisions) / len(precisions) if precisions else 0.0

def get_F1(SR, GT, num_classes=4, background_label=0):
    SR_classes = torch.argmax(SR, dim=1)
    f1_scores = []
    for class_label in range(num_classes):
        if class_label == background_label:
            continue
        SR_class = (SR_classes == class_label).float()
        GT_class = (GT == class_label).float()
        TP = ((SR_class == 1) & (GT_class == 1)).sum()
        FP = ((SR_class == 1) & (GT_class == 0)).sum()
        FN = ((SR_class == 0) & (GT_class == 1)).sum()
        PC = float(TP) / (float(TP + FP) + 1e-6)
        SE = float(TP) / (float(TP + FN) + 1e-6)
        F1 = 2 * SE * PC / (SE + PC + 1e-6)
        f1_scores.append(F1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def get_JS(SR_pred, GT, class_label):
    SR_class = SR_pred == class_label
    GT_class = GT == class_label
    Inter = torch.sum(SR_class & GT_class).item()
    Union = torch.sum(SR_class | GT_class).item()
    JS = float(Inter) / (float(Union) + 1e-6)
    return JS

def get_DC(SR_pred, GT, class_label):
    SR_class = SR_pred == class_label
    GT_class = GT == class_label
    Inter = torch.sum(SR_class & GT_class).item()
    DC = float(2 * Inter) / (float(torch.sum(SR_class) + torch.sum(GT_class)) + 1e-6)
    return DC