import cv2 as cv
import os
import numpy as np
import pickle
from sklearn import svm
import random as rng


rng.seed(12345)
threshold = 100


def hist_feature(img_file, bins):
    """
    Returns histogram of an input image
    :param img_file: path to img_file
    :return: histogram as feature vector
    """
    img = cv.imread(img_file, 1)
    hist, bins = np.histogram(img.flatten(), bins=bins, range=[0, 255])
    return hist


def feature_extract(directory):
    """
    Extract all the feature and label them for each image group in the relevant forlder.
    :param directory:
    :return:
    """
    label_list = []
    features_list = []
    for root, dirs, files in os.walk(directory):
        print('directory is {}'.format(dirs))
        print('root is {}'.format(root))
        for d in dirs:
            print(d)
            images = os.listdir(os.path.join(root, d))
            for image in images:
                label_list.append(d)
                # feature_vector = pixel_feature( os.path.join(root, d) + '/' + image)
                feature_vector = hist_feature(os.path.join(root, d) + '/' + image, bins=6)
                features_list.append(feature_vector)
    return (np.asarray(features_list), np.asarray(label_list))


def apply_classifier(classifier, frame):
    """
    Loads in a trained classifier and apply on input frame
    :param classifier: any trained classifier. In this case it is either SVM or KNN classifiers.
    :param frame: input frame to apply the classifier.
    :return: Color of the input frame (car's color)
    """
    # loaded_model = pickle.load(open('svm_color_classifier.pkl', 'rb'))
    loaded_model = pickle.load(open(classifier, 'rb'))
    hist, bins = np.histogram(frame.flatten(), bins=9, range=[0, 255])
    result = loaded_model.predict(hist)
    return result


def extract_features_hist(frame, bins):
    """
    Calculates histogram of input color image
    :param img_path: full absolute path to image
    :param bins: number of bins to calculate histogram
    :return: histogoram as list
    """
    imag = frame
    hist, bins = np.histogram(imag, bins=bins, range=[0, 255])
    return list(hist.squeeze())


def apply_classifier_svm(frame):
    """
    Loads in a trained classifier and apply on input frame
    :param classifier: any trained classifier. In this case it is either SVM or KNN classifiers.
    :param frame: input frame to apply the classifier.
    :return: Color of the input frame (car's color)
    """
    # loaded_model = pickle.load(open('svm_color_classifier.pkl', 'rb')) # bins = 9
    loaded_model = pickle.load(open('svm_color_classifier_poly.pkl', 'rb')) # bins = 24
    hist = extract_features_hist(frame, 24)
    result = loaded_model.predict(np.array(hist).reshape(1, -1))
    return result


def draw_contor(src_gray, thresh):
    threshold = thresh
    frame_input = src_gray.copy()
    src_gray = cv.cvtColor(src_gray, cv.COLOR_BGR2GRAY)
    src_gray = cv.medianBlur(src_gray, 3)
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    mask = np.zeros(frame_input.shape[:2], np.uint8)
    mask_tmp = np.zeros(frame_input.shape[:2], np.uint8)
    for i, contour in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(contour, 3, True)
        area = cv.contourArea(contours_poly[i])
        boundRect[i] = cv.boundingRect(contours_poly[i])
        (x, y, w, h) = boundRect[i]
        if ( 50 < area ) & (area < 500) & (radius[i] < 60) & (radius[i] > 31) & (h <= 70):
            print('Width is {}'.format(w))
            print('Height is {}'.format(h))
            # Here is where the classifier should apply
            mask_tmp[y:y + h, x:x + w] = 255
            predicted_contour = cv.bitwise_and(frame_input, frame_input, mask=mask_tmp)
            print('Predicted contour shape {}'.format(predicted_contour.shape))
            car_feature = extract_features_hist(predicted_contour, 24) # p for the other classifier
            predicted_color = apply_classifier_svm(car_feature)
            print('Predicted color is {}'.format(predicted_color[0]))
            # mask is only for imshow purposes
            mask = mask + mask_tmp
            cv.drawContours(drawing, contours_poly, i, [120, 120, 50])
            cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), [0, 255, 0], 1)
            cv.rectangle(frame_input, (x, y), (x + w, y + h), [0, 0, 255], 2)

            # Put the label on each rectangle
            font = cv.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            # cv.putText(mask_tmp, 'Hello World!',
            #             bottomLeftCornerOfText,
            #             font,
            #             fontScale,
            #             fontColor,
            #             lineType)
            cv.putText(frame_input, predicted_color[0], (x, y - 5), font, .5, (255, 255, 255), 2)

    res = cv.bitwise_and(frame_input, frame_input, mask=mask)
    cv.imshow('Contours', drawing)
    # cv.imshow('mask', mask)
    cv.imshow('Bounding rects on frame', frame_input)
    cv.imshow('masked frame', res)
    # cv.waitKey(0)

if __name__ == '__main__':
    # Read a video in
    video_path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses5\Ex_4'
    video_name = '1.mp4'
    vid_path = os.path.join(video_path, video_name)
    cap = cv.VideoCapture(vid_path)

    try:
        while(cap.isOpened()):
            ret,frame = cap.read()

            if ret == True:
                cv.imshow('video of street', frame)
                draw_contor(frame, 100)

                if cv.waitKey(25) == ord('q'):
                    break
            else:
                break

    except Exception as e:
        print(e)
        raise

    finally:
        # release the video capture object
        cap.release()
        cv.destroyAllWindows()
