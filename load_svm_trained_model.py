# load svm trained model to predict cars color
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
import os
import cv2 as cv
import numpy as np
from sklearn.metrics import plot_confusion_matrix


def extract_features_hist(img_path, bins):
    """
    Calculates histogram of input color image
    :param img_path: full absolute path to image
    :param bins: number of bins to calculate histogram
    :return: histogoram as list
    """
    imag = cv.imread(img_path, 1)
    hist, bins = np.histogram(imag.flatten(), bins=bins, range=[0, 255])
    return list(hist)
    # return list(hist.squeeze())


def scan_folder(path):
    '''
    extracts files from a specific folder
    :param path: input path of a folder
    :return: all images in the path as a list
    '''
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


if __name__ == "__main__":
    src_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\cars'
    dir_list = os.listdir(src_path)
    dest_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses3\HW_Ses3\kmean_cars'
    labels = []

    biin = 24
    print('Running to save best model for bin = {}'.format(biin))
    # calculate histogram (extract features for each car)
    # take the number of features the same as best number of centroids + 2
    dir_list = os.listdir(src_path)
    labels = []
    feature_vector = []
    for fol in dir_list:
        labels.append(fol)
        n_path = os.path.join(src_path, fol)
        img_list = scan_folder(n_path)
        for img in img_list:
            car_feature = extract_features_hist(os.path.join(n_path, img), biin)
            car_feature.insert(0, fol)
            feature_vector.append(car_feature)

    y = [row[0] for row in feature_vector]
    X = [row[1:] for row in feature_vector]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

    # loaded_model = pickle.load(open('svm_color_classifier.pkl', 'rb'))
    loaded_model = pickle.load(open('svm_color_classifier_poly.pkl', 'rb'))
    result = loaded_model.score(X_test, y_test)
    print('X_test index 0 is {}'.format(X_test[0]))
    print('result is {}'.format(result))

    # pred = loaded_model.predict(np.array([80945, 115532, 228628, 284049, 246331, 234232, 193999, 149803, 176310]).reshape(1, -1))
    #     # ([3, 0, 0, 0, 1, 0, 0, 2, 0])
    # print('pred is {}'.format(pred))

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(loaded_model, X_test, y_test,
                                     display_labels=labels,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

# svm_color_classifier_poly.pkl Running to save the best model for bin = 24 score=62.5%
# svm_color_classifier_sigmoid.pkl Running to save the best model for bin = 7 score = 40%
# svm_color_classifier_rbf.pkl Running to save the best model for bin = 24 score = 50%
# svm_color_classifier_poly_gamma01.pkl[{'bins': 14, 'score': 0.55}] Running to save the best model for bin = 14 g=0.1
# C=100, gamma = 1, poly score = 50% Running to save the best model for bin = 9












