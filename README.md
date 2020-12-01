# object_detection_by_active_contours_Canny_algorithm

. Using Canny and active contour algorithms to detect cars on a street. Thresholding has been used to detect
the right size of the contour to show.
. A trained SVM model is later applied on the contour area to predict the color of the cars. The SVM model
is in the repository as pickle file. It has been trained on histogram of kmeans of cars with 9 bins for the histogram.
