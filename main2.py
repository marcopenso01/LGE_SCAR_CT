import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# for GPU process:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn import metrics
import cv2
import h5py
import pandas as pd
import glob
import tensorflow as tf
import model_zoo as model_zoo
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from packaging import version
from tensorflow.python.client import device_lib

assert 'GPU' in str(device_lib.list_local_devices())

print('is_gpu_available: %s' % tf.test.is_gpu_available())  # True/False
# Or only check for gpu's with cuda support
print('gpu with cuda support: %s' % tf.test.is_gpu_available(cuda_only=True))
# tf.config.list_physical_devices('GPU') #The above function is deprecated in tensorflow > 2.1

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "this notebook requires Tensorflow 2.0 or above"


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def print_txt(output_dir, stringa):
    out_file = os.path.join(output_dir, 'summary_report.txt')
    with open(out_file, "a") as text_file:
        text_file.writelines(stringa)


directory = 'D:\TAC_scar\data8'
for f in glob.glob(os.path.join(directory, "*.hdf5")):
    f = f.split("/")[-1]
    fold = f.split("_")[-1].split(".")[0]
    print('-' * 70)
    print('Pre-processing file: %s' % f)
    print('-' * 70)

    fold_name = 'cnn'
    output_folder = os.path.join(directory, fold_name, fold)
    if not os.path.exists(output_folder):
        makefolder(output_folder)
        out_file = os.path.join(output_folder, 'summary_report.txt')
        with open(out_file, "w") as text_file:
            text_file.write('\n\n--------------------------------------------------------------------------\n')
            text_file.write('Model summary\n')
            text_file.write('----------------------------------------------------------------------------\n\n')

    data = h5py.File(os.path.join(directory, f), 'r')
    train_images = data['segs_tr'][()]
    train_labels = data['out_tr'][()]
    test_images = data['segs_test'][()]
    test_labels = data['out_test'][()]
    test_patient = data['paz_test'][()]
    data.close()
    print('training data', len(train_images), train_images[0].shape)
    print('testing data', len(test_images), test_images[0].shape)
    print_txt(output_folder, ['\ntraining data %d' % len(train_images)])
    print_txt(output_folder, ['\ntest data %d\n\n' % len(test_images)])

    img_size = train_images[0].shape[0]

    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    train_datagen = ImageDataGenerator(
        rotation_range=360,
        #zoom_range=0.1,
        #width_shift_range=0.10,
        #height_shift_range=0.10,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

    test_datagen = ImageDataGenerator()

    batch_size = 16
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
    valid_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)
    # test_generator = test_datagen.flow(test_images, batch_size=1)

    callbacks = [EarlyStopping(monitor='val_loss', patience=16, verbose=1), \
                 ReduceLROnPlateau(factor=0.2, patience=6, min_lr=0.000001, verbose=1), \
                 ModelCheckpoint(os.path.join(output_folder, 'model.h5'), monitor="val_loss", verbose=1, save_best_only=True, \
                                 save_weights_only=False, mode='min')]

    model = model_zoo.get_model()
    # model.summary()
    with open(out_file, "a") as text_file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

    opt = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',
                                                                      tf.keras.metrics.AUC(),
                                                                      tf.keras.metrics.Recall(),
                                                                      tf.keras.metrics.Precision()])
    print('model prepared...')
    print('training started...')
    results = model.fit(train_generator, epochs=200, validation_data=valid_generator, verbose=2, callbacks=callbacks)
    print('Model correctly trained and saved')

    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plt.title("Learning curve LOSS", fontsize=20)
    plt.plot(results.history["loss"], label="Loss")
    plt.plot(results.history["val_loss"], label="Validation loss")
    p = np.argmin(results.history["val_loss"])
    plt.plot(p, results.history["val_loss"][p], marker="x", color="r", label="best model")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend();
    plt.savefig(os.path.join(output_folder, 'Loss'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plt.title("Learning curve ACCURACY", fontsize=20)
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="Validation Accuracy")
    plt.plot(p, results.history["val_accuracy"][p], marker="x", color="r", label="best model")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend();
    plt.savefig(os.path.join(output_folder, 'Accuracy'), dpi=300)
    plt.close()

    # free memory
    del train_images
    del train_labels

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    TESTING AND EVALUATING THE MODEL
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print('-' * 50)
    print('Testing')
    print('Testing data', len(test_images), test_images[0].shape)
    print_txt(output_folder, ['\nTesting data %d' % len(test_images)])
    print('Loading saved weights...')
    model = tf.keras.models.load_model(os.path.join(output_folder, 'model.h5'))
    prediction = model.predict(test_images)

    # calculate roc curves
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, prediction, pos_label=1)
    aucc = metrics.roc_auc_score(test_labels, prediction)
    # plot the roc curve for the model
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label="ROC curve (area = %0.2f)" % aucc)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    # plt.show()
    plt.savefig(os.path.join(output_folder, 'AUC'), dpi=300)
    plt.close()

    # define thresholds
    thresholds = np.arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [metrics.f1_score(test_labels, to_labels(prediction, t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    # print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    print_txt(output_folder, ['\nThreshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix])])
    pred_adj = adjusted_classes(prediction, thresholds[ix])
    # precision
    precision = metrics.precision_score(test_labels, pred_adj)
    # print('Precision: %.2f' % precision)
    print_txt(output_folder, ['\nPrecision: %.2f' % precision])
    # recall
    recall = metrics.recall_score(test_labels, pred_adj)
    # print('Recall: %.2f' % recall)
    print_txt(output_folder, ['\nRecall: %.2f' % recall])
    # f1
    f1 = metrics.f1_score(test_labels, pred_adj)
    # print('f1: %.2f' % f1)
    print_txt(output_folder, ['\nf1: %.2f' % f1])
    # ROC AUC
    print('ROC AUC: %f' % aucc)
    print_txt(output_folder, ['\nROC AUC: %f' % aucc])

    # print(metrics.classification_report(test_labels, pred_adj))
    print_txt(output_folder, ['\n\n %s \n\n' % metrics.classification_report(test_labels, pred_adj)])

    CM = metrics.confusion_matrix(test_labels, pred_adj)
    disp = metrics.ConfusionMatrixDisplay(CM)
    disp.plot()
    # plt.show()
    plt.savefig(os.path.join(output_folder, 'Conf_matrix'), dpi=300)
    plt.close()

    TN = CM[0][0]
    # print('true negative:', TN)
    print_txt(output_folder, ['\ntrue negative: %d' % TN])
    FN = CM[1][0]
    # print('false negative:', FN)
    print_txt(output_folder, ['\nfalse negative: %d' % FN])
    TP = CM[1][1]
    # print('true positive:', TP)
    print_txt(output_folder, ['\ntrue positive: %d' % TP])
    FP = CM[0][1]
    # print('false positive:', FP)
    print_txt(output_folder, ['\nfalse positive: %d' % FP])
    # print('Precision or Pos predictive value: %.2f' % (TP/(TP+FP)))
    print_txt(output_folder, ['\nPrecision or Pos predictive value: %.2f' % (TP / (TP + FP))])
    # print('Recall: %.2f' % (TP/(TP+FN)))
    print_txt(output_folder, ['\nRecall: %.2f' % (TP / (TP + FN))])
    # print('Specificity: %.2f' % (TN/(TN+FP)))
    print_txt(output_folder, ['\nSpecificity: %.2f' % (TN / (TN + FP))])
    # print('Neg predictive value: %.2f' % (TN/(FN+TN)))
    print_txt(output_folder, ['\nNeg predictive value: %.2f' % (TN / (FN + TN))])
    # accuracy:
    print('accuracy: %d' % ((TP+TN)/(TN+FN+TP+FP)))
    print_txt(output_folder, ['\nAccuracy: %d\n\n' % ((TP+TN)/(TN+FN+TP+FP))])

    with open(out_file, "a") as text_file:
        text_file.write('\n----- Prediction ----- \n')
        text_file.write('patient       pred_class       probability\n')
        for ii in range(len(prediction)):
            text_file.write(
                '%d             %d                %.3f\n' % (test_patient[ii], test_labels[ii], prediction[ii]))


    list_paz = np.unique(test_patient)
    for pp in range(len(list_paz)):
        print('\ntesting patient %d' % list_paz[pp])
        print_txt(output_folder, ['\ntesting patient %d' % list_paz[pp]])
        temp_data = []
        temp_out = []
        for ii in range(len(test_images)):
            if test_patient[ii] == list_paz[pp]:
                temp_data.append(test_images[ii])
                temp_out.append(test_labels[ii])
        temp_data = np.asarray(temp_data)
        temp_out = np.asarray(temp_out)
        prediction = model.predict(temp_data)
        print('ROC AUC: %f' % metrics.roc_auc_score(temp_out, prediction))
        print_txt(output_folder, ['\nROC AUC: %f' % metrics.roc_auc_score(temp_out, prediction)])

    paz = []
    lab = []
    pred = []
    for ii in range(len(prediction)):
        paz.append(test_patient[ii])
        lab.append(test_labels[ii])
        pred.append(prediction[ii])
    df1 = pd.DataFrame({'paz': paz, 'lab': lab, 'pred': pred})
    df1.to_excel(os.path.join(directory, fold_name, 'Excel_df1.xlsx'))

    del test_images
    del test_labels
    del test_patient
