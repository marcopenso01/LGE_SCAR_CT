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
