import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import os

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

df0 = pd.read_excel(os.path.join('D:\TAC_scar\data7','file0.xlsx'),
     engine='openpyxl',
)
df1 = pd.read_excel(os.path.join('D:\TAC_scar\data7','file1.xlsx'),
     engine='openpyxl',
)
df2 = pd.read_excel(os.path.join('D:\TAC_scar\data7','file2.xlsx'),
     engine='openpyxl',
)
df3 = pd.read_excel(os.path.join('D:\TAC_scar\data7','file3.xlsx'),
     engine='openpyxl',
)
df4 = pd.read_excel(os.path.join('D:\TAC_scar\data7','file4.xlsx'),
     engine='openpyxl',
)

lab0 = np.asarray(df0['lab'])
pred0 = np.asarray(df0['pred'])
lab1 = np.asarray(df1['lab'])
pred1 = np.asarray(df1['pred'])
lab2 = np.asarray(df2['lab'])
pred2 = np.asarray(df2['pred'])
lab3 = np.asarray(df3['lab'])
pred3 = np.asarray(df3['pred'])
lab4 = np.asarray(df4['lab'])
pred4 = np.asarray(df4['pred'])

aucc0 = metrics.roc_auc_score(lab0, pred0)
aucc1 = metrics.roc_auc_score(lab1, pred1)
aucc2 = metrics.roc_auc_score(lab2, pred2)
aucc3 = metrics.roc_auc_score(lab3, pred3)
aucc4 = metrics.roc_auc_score(lab4, pred4)

print(aucc0, aucc1, aucc2, aucc3, aucc4)

fpr0, tpr0, thresholds0 = metrics.roc_curve(lab0, pred0, pos_label=1)
fpr1, tpr1, thresholds1 = metrics.roc_curve(lab1, pred1, pos_label=1)
fpr2, tpr2, thresholds2 = metrics.roc_curve(lab2, pred2, pos_label=1)
fpr3, tpr3, thresholds3 = metrics.roc_curve(lab3, pred3, pos_label=1)
fpr4, tpr4, thresholds4 = metrics.roc_curve(lab4, pred4, pos_label=1)

mean_auc = np.mean([aucc0, aucc1, aucc2, aucc3, aucc4])
std_auc = np.std([aucc0, aucc1, aucc2, aucc3, aucc4])
infer_CI = mean_auc - (1.96 * std_auc / np.sqrt(5))
upper_CI = mean_auc + (1.96 * std_auc / np.sqrt(5))

plt.plot([0,1], [0,1], linestyle='--', lw=2, color="r", alpha=0.8)
plt.plot(fpr0, tpr0, lw=2, alpha=0.5, label="ROC fold 0 (AUC = %0.2f)" % aucc0)
plt.plot(fpr1, tpr1, lw=2, alpha=0.5, label="ROC fold 1 (AUC = %0.2f)" % aucc1)
plt.plot(fpr2, tpr2, lw=2, alpha=0.5, label="ROC fold 2 (AUC = %0.2f)" % aucc2)
plt.plot(fpr3, tpr3, lw=2, alpha=0.5, label="ROC fold 3 (AUC = %0.2f)" % aucc3)
plt.plot(fpr4, tpr4, lw=2, alpha=0.5, label="ROC fold 4 (AUC = %0.2f)" % aucc4)
plt.plot([], [], ' ', label="Mean AUC = %0.2f (CI %0.2f-%0.2f)" % (mean_auc,infer_CI,upper_CI))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.savefig(os.path.join('D:\TAC_scar\data7', 'AUC'), dpi=1200)
plt.close()

scores = [metrics.f1_score(lab0, to_labels(pred0, t)) for t in thresholds0]
ix = np.argmax(scores)
pred_adj0 = adjusted_classes(pred0, thresholds0[ix])
CM0 = metrics.confusion_matrix(lab0, pred_adj0)

scores = [metrics.f1_score(lab1, to_labels(pred1, t)) for t in thresholds1]
ix = np.argmax(scores)
pred_adj1 = adjusted_classes(pred1, thresholds1[ix])
CM1 = metrics.confusion_matrix(lab1, pred_adj1)

scores = [metrics.f1_score(lab2, to_labels(pred2, t)) for t in thresholds2]
ix = np.argmax(scores)
pred_adj2 = adjusted_classes(pred2, thresholds2[ix])
CM2 = metrics.confusion_matrix(lab2, pred_adj2)

scores = [metrics.f1_score(lab3, to_labels(pred3, t)) for t in thresholds3]
ix = np.argmax(scores)
pred_adj3 = adjusted_classes(pred3, thresholds3[ix])
CM3 = metrics.confusion_matrix(lab3, pred_adj3)

scores = [metrics.f1_score(lab4, to_labels(pred4, t)) for t in thresholds4]
ix = np.argmax(scores)
pred_adj4 = adjusted_classes(pred4, thresholds4[ix])
CM4 = metrics.confusion_matrix(lab4, pred_adj4)

TN = CM0[0][0]
FN = CM0[1][0]
TP = CM0[1][1]
FP = CM0[0][1]
PPV1 = (TP / (TP + FP))
NPV1 = (TN / (FN + TN))
ACC1 = (TP+TN)/(TN+FN+TP+FP)
recall1 = (TP / (TP + FN))
TN = CM1[0][0]
FN = CM1[1][0]
TP = CM1[1][1]
FP = CM1[0][1]
PPV2 = (TP / (TP + FP))
NPV2 = (TN / (FN + TN))
ACC2 = (TP+TN)/(TN+FN+TP+FP)
recall2 = (TP / (TP + FN))
TN = CM2[0][0]
FN = CM2[1][0]
TP = CM2[1][1]
FP = CM2[0][1]
PPV3 = (TP / (TP + FP))
NPV3 = (TN / (FN + TN))
ACC3 = (TP+TN)/(TN+FN+TP+FP)
recall3 = (TP / (TP + FN))
TN = CM3[0][0]
FN = CM3[1][0]
TP = CM3[1][1]
FP = CM3[0][1]
PPV4 = (TP / (TP + FP))
NPV4 = (TN / (FN + TN))
ACC4 = (TP+TN)/(TN+FN+TP+FP)
recall4 = (TP / (TP + FN))
TN = CM4[0][0]
FN = CM4[1][0]
TP = CM4[1][1]
FP = CM4[0][1]
PPV5 = (TP / (TP + FP))
NPV5 = (TN / (FN + TN))
ACC5 = (TP+TN)/(TN+FN+TP+FP)
recall5 = (TP / (TP + FN))

mean_ppv = np.mean([PPV1, PPV2, PPV3, PPV4, PPV5])
mean_npv = np.mean([NPV1, NPV2, NPV3, NPV4, NPV5])
mean_acc = np.mean([ACC1, ACC2, ACC3, ACC4, ACC5])
mean_recall = np.mean([recall1, recall2, recall3, recall4, recall5])
std_ppv = np.std([PPV1, PPV2, PPV3, PPV4, PPV5])
std_npv = np.std([NPV1, NPV2, NPV3, NPV4, NPV5])
std_acc = np.std([ACC1, ACC2, ACC3, ACC4, ACC5])
std_recall = np.std([recall1, recall2, recall3, recall4, recall5])

infer_ppv = mean_ppv - (1.96 * std_ppv / np.sqrt(5))
upper_ppv = mean_ppv + (1.96 * std_ppv / np.sqrt(5))
infer_npv = mean_npv - (1.96 * std_npv / np.sqrt(5))
upper_npv = mean_npv + (1.96 * std_npv / np.sqrt(5))
infer_acc = mean_acc - (1.96 * std_acc / np.sqrt(5))
upper_acc = mean_acc + (1.96 * std_acc / np.sqrt(5))
infer_recall = mean_recall - (1.96 * std_recall / np.sqrt(5))
upper_recall = mean_recall + (1.96 * std_recall / np.sqrt(5))

print("Mean PPV = %0.2f (CI %0.2f-%0.2f)" % (mean_ppv,infer_ppv,upper_ppv))
print("Mean NPV = %0.2f (CI %0.2f-%0.2f)" % (mean_npv,infer_npv,upper_npv))
print("Mean ACC = %0.2f (CI %0.2f-%0.2f)" % (mean_acc,infer_acc,upper_acc))
print("Mean RECALL = %0.2f (CI %0.2f-%0.2f)" % (mean_recall,infer_recall,upper_recall))
print("Mean AUC = %0.2f (CI %0.2f-%0.2f)" % (mean_auc,infer_CI,upper_CI))
