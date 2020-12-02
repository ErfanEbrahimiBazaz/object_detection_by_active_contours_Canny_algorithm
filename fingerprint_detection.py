# detect the finger print by edge detection in each image
import cv2 as cv
import numpy as np
import os


# model variables
canny_threshold = 100
if __name__ == "__main__":
    path = r'C:\Users\E17538\OneDrive - Uniper SE\Desktop\DailyActivities\FAD\ACV_Ses5\Ex_4\FVC2002\DB1_B'
    for root, dirs, files in os.walk(path):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv.imread(img_path, 1)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_blur = cv.bilateralFilter(img_gray, 5, 150, 150)
            # img_blur = cv.GaussianBlur(img_bilat, (7,7), 25)
            img_blur = cv.blur(img_blur, (5, 5))
            img_med_blur = cv.medianBlur(img_blur, 7)

            canny_output = cv.Canny(img_blur, canny_threshold, canny_threshold * 2)
            contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            contour_poly = [None] * len(contours)
            bound_rect = [None] * len(contours)

            height, width = canny_output.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            drawing = np.zeros((canny_output.shape[:2]), dtype=np.uint8)

            for i, contour in enumerate(contours):
                contour_poly[i] = cv.approxPolyDP(contour, 10, True)
                bound_rect[i] = cv.boundingRect(contour_poly[i])
                # computes the bounding box for the contour, and draws it on the frame,
                (x, y, w, h) = bound_rect[i]
                if (w > 50) & (h > 50):
                    cv.rectangle(img, (x, y), (x + w, y + h), [0, 0, 255], 2)
                    # drawing: the source image, contour_poly: the contour to draw, i: idx of contour, []: color_vector
                    # cv.drawContours(drawing, contour_poly, i, [120, 120, 0])

            # -1 in cv.drawContours() is to draw all contours
            cv.drawContours(drawing, contour_poly, -1, [120, 120, 0])
            cv.imshow('finger print gray', img_gray)
            cv.imshow('Canny output', canny_output)
            cv.imshow('finger print', img)
            cv.imshow('Contours', drawing)
            # cv.imshow('contour', np.array(contour, np.int32))
            if cv.waitKey(3000) == 'q':
                break


cv.destroyAllWindows()


