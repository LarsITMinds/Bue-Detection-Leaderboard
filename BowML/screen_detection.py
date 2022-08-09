from PIL import Image
import numpy as np
import cv2


def find_scoreboard_screen(images: [[]]):
    count = 0
    black_threshold = 45
    resize_factor = 4

    print(len(images))

    for imageArray in images:
        image = imageArray[0].copy()
        count += 1

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (thresh, blackAndWhiteImage) = cv2.threshold(gray_image, black_threshold, 255, cv2.THRESH_BINARY)
        resized_image = cv2.resize(blackAndWhiteImage, (int(blackAndWhiteImage.shape[1]/resize_factor), int(blackAndWhiteImage.shape[0]/resize_factor)))
        kernel = np.ones((5, 5), np.uint8)

        #cv2.imshow(imageArray[1], resized_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        closing = cv2.morphologyEx(resized_image, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        a = 0, 0
        b = 0, 0
        c = 0, 0
        d = 0, 0
        e = 0, 0
        for i in range(0, len(contours)):
            area_size = cv2.contourArea(contours[i])
            if area_size > a[0]:
                e = d
                d = c
                c = b
                b = a
                a = area_size, i
            elif area_size > b[0]:
                e = d
                d = c
                c = b
                b = area_size, i
            elif area_size > c[0]:
                e = d
                d = c
                c = area_size, i
            elif area_size > d[0]:
                e = d
                d = area_size, i
            elif area_size > e[0]:
                e = area_size, i

        resized_original_image = cv2.resize(image, (int(image.shape[1]/resize_factor), int(image.shape[0]/resize_factor)))
        cropped_image = crop_and_rotate_image_by_contours(contours[b[1]]*resize_factor, image)

        box_image1 = draw_box_in_image(contours[a[1]]*resize_factor, image)
        box_image2 = draw_box_in_image(contours[b[1]]*resize_factor, box_image1)
        box_image3 = draw_box_in_image(contours[c[1]]*resize_factor, box_image2)
        box_image4 = draw_box_in_image(contours[d[1]]*resize_factor, box_image3)
        box_image5 = draw_box_in_image(contours[e[1]]*resize_factor, box_image4)

        # cv2.imshow('Boxing image', box_image5)
        # cv2.imshow('Cropped image', cropped_image)

        cv2.imwrite('Test data/Screen detection results/' + imageArray[1] + '.jpg', cropped_image)
        cv2.imwrite('Test data/Black & White results/' + imageArray[1] + '.jpg', closing)
        cv2.imwrite('Test data/Boxing results/' + imageArray[1] + '.jpg', box_image5)


def draw_box_in_image(cnt, image):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return cv2.drawContours(image, [box], 0, (0, 255, 0), 8)


def crop_and_rotate_image_by_contours(cnt, image: Image) -> Image:
    rect = cv2.minAreaRect(cnt)

    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = image.shape[0], image.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    rotated_image = cv2.warpAffine(image, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    return cv2.getRectSubPix(rotated_image, size, center)
