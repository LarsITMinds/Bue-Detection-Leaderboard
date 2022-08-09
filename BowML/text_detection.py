from PIL import Image
import cv2
import numpy as np


contour_border_margin = 0.5  # The number is percent of the original image given to the length of the contours border

def find_texts_on_screen(original_image: Image, file_name: str, should_create_pictures: bool):
    image = original_image.copy()

    no_brownish_image = remove_brownish_color(image, file_name, should_create_pictures)
    gray_image = cv2.cvtColor(no_brownish_image, cv2.COLOR_BGR2GRAY)
    (_, bw2) = cv2.threshold(gray_image, 155, 255.0, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, kernel)

    grad = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))

    rough_connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


    erosion_kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(rough_connected, erosion_kernel, iterations=1)
    dilate = cv2.dilate(erosion, erosion_kernel, iterations=1)

    connected = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('Image', erosion)
    #cv2.imshow('Image', erosion)
    #cv2.imshow('Original image', connected)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)

    possible_text_areas = []
    for index in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[index])
        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, index, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
        if does_contour_meet_the_criteria(image, x - int(), y, r, w, h):
            border_addition = original_image.shape[1] * contour_border_margin / 100
            new_x = int(x - border_addition)
            new_width = int(w + border_addition * 2)

            possible_text_areas.append([new_x, y, new_width, h])

            #if should_create_pictures:
            #    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

    #### Combine possible contours ####
    #print(len(possible_text_areas))
    possible_text_areas = look_for_and_combine_possible_contours(possible_text_areas, 0, image.shape[0],
                                                                 image.shape[1])
    #print(len(possible_text_areas))
    ##########################

    if should_create_pictures:
        for area in possible_text_areas:
            x = area[0]
            y = area[1]
            w = area[2]
            h = area[3]
            cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)




    if should_create_pictures:
        cv2.imwrite('Score results/Images/Connection images/' + file_name + ' contours.jpg', image)
        cv2.imwrite('Score results/Images/Connection images/' + file_name + ' connected.jpg', connected)
        cv2.imwrite('Score results/Images/Connection images/' + file_name + ' rough_connected.jpg', rough_connected)
        cv2.imwrite('Score results/Images/Connection images/' + file_name + ' erosion.jpg', erosion)
        cv2.imwrite('Score results/Images/Connection images/' + file_name + ' dilate.jpg', dilate)
        cv2.imwrite('Score results/Images/Connection images/' + file_name + ' bw.jpg', bw)

    return possible_text_areas


def remove_brownish_color(image: Image, file_name: str, should_create_pictures: bool) -> Image:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brown_lo = np.array([10, 0, 0])
    brown_hi = np.array([75, 255, 255])

    # Mask image to only select brownish
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to black where we found brownish
    image[mask > 0] = (0, 0, 0)

    if should_create_pictures:
        cv2.imwrite('Score results/Images/HSV images/' + file_name + ' hsv.jpg', hsv)
        cv2.imwrite('Score results/Images/HSV images/' + file_name + '.jpg', image)

    return image


def look_for_and_combine_possible_contours(
        contour_list: [],
        index_to_start_from: int,
        image_height: int,
        image_width: int) -> []:
    #print(contour_list)
    #print(image_height)
    #print(image_width)
    new_contour_list = contour_list.copy()

    if index_to_start_from >= len(contour_list):
        return contour_list

    height_threshold = image_height * 0.01
    width_threshold = image_width * 0.02

    for outer_index in range(index_to_start_from, len(contour_list)):
        if outer_index + 1 >= len(contour_list):
            return contour_list

        x1, y1, w1, h1 = contour_list[outer_index]

        for inner_index in range(outer_index + 1, len(contour_list)):
            x2, y2, w2, h2 = contour_list[inner_index]
            y_difference = abs(y1 - y2)
            if y_difference < height_threshold:
                first_x_distance = abs(x1 - (x2 + w2))
                second_x_distance = abs(x2 - (x1 + w1))

                if first_x_distance < width_threshold or second_x_distance < width_threshold:
                    #print("Possible merge:")
                    #print(contour_list[outer_index])
                    #print(contour_list[inner_index])
                    #print()

                    new_width = x1 + w1 - x2 if x1 > x2 else x2 + w2 - x1
                    new_height = y1 + h1 - y2 if y1 > y2 else y2 + h2 - y1

                    new_contour = [min(x1, x2), min(y1, y2), new_width, new_height]
                    new_contour_list[outer_index] = new_contour
                    del new_contour_list[inner_index]

                    return look_for_and_combine_possible_contours(new_contour_list, outer_index, image_height, image_width)

    return new_contour_list

def look_for_and_combine_possible_words(
        contour_list: [[], [int, int, int, int]],
        image_height,
        image_width) -> []:

    possible_words = []
    found_combinations = 0

    height_threshold = image_height * 0.01
    width_threshold = image_width * 0.10

    for outer_index in range(len(contour_list)):
        if outer_index >= len(contour_list):
            break

        word_section = contour_list[outer_index]
        word_position = word_section[1]
        possible_word = ''

        for symbol_group in word_section[0]:
            #print(symbol_group)
            for symbol in symbol_group:
                #print(symbol)
                possible_word += symbol[1]
        #print(possible_word)


        for index in range(len(contour_list)):
            new_index = outer_index + index + 1
            #print("index: " + str(new_index))
            #print("length: " + str(len(contour_list)))
            if new_index >= len(contour_list):
                possible_words.append([possible_word, word_position])
                break

            x1, y1, w1, h1 = word_position
            x2, y2, w2, h2 = contour_list[new_index][1]

            first_y_distance = abs(y1 - (y2 + h2))
            second_y_distance = abs(y2 - (y1 + h1))
            x_distance = abs(x1 - x2)


            if (first_y_distance < height_threshold or second_y_distance < height_threshold) and x_distance < width_threshold:
                word_to_compare = ''
                for symbol_group in contour_list[new_index][0]:
                    for symbol in symbol_group:
                        word_to_compare += symbol[1]

                #print("Possible merge:")
                #print(possible_word)
                #print(word_to_compare)
                #print(x_distance)
                #print(width_threshold)
                #print()

                new_x = x1 if x1 < x2 else x2
                new_y = y1 if y1 < y2 else y2
                new_width = w1 if w1 > w2 else w2
                new_height = y1 + h1 - y2 if first_y_distance < height_threshold else y2 + h2 - y1

                if y1 < y2:
                    possible_words.append([possible_word + ' ' + word_to_compare, [new_x, new_y, new_width, new_height]])
                else:
                    possible_words.append([word_to_compare + ' ' + possible_word, [new_x, new_y, new_width, new_height]])

                found_combinations = found_combinations + 1
                del contour_list[new_index]
                break

    return possible_words

def does_contour_meet_the_criteria(
        image: Image,
        x: int,
        y: int,
        ratio: float,
        width: int,
        height: int) -> bool:

    if ratio > 0.30 and width > 30 and 20 < height < 200:
        image_height = image.shape[0]
        image_width = image.shape[1]

        x_lower_threshold = 0.10
        x_higher_threshold = 0.90
        y_lower_threshold = 0.30
        y_higher_threshold = 0.90

        is_x_within_boundaries = image_width * x_lower_threshold < x < image_width * x_higher_threshold
        is_y_within_boundaries = image_height * y_lower_threshold < y < image_height * y_higher_threshold

        #if is_x_within_boundaries and is_y_within_boundaries:
        return True

    return False
