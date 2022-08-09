import numpy as np
import cv2


def get_training_data(images: []):
    result_of_image = ''
    current_training_image_set = []

    image_training_values = {}

    for image_data in images:
        image = cv2.resize(image_data[0], (50, 50))
        file_name = image_data[1]

        file_name_split = file_name.split(' ')

        if result_of_image == '':
            result_of_image = file_name_split[0]
            current_training_image_set.append(image)
        elif result_of_image != file_name_split[0]:
            image_training_values[result_of_image] = build_training_data_for_image_set(current_training_image_set)
            result_of_image = file_name_split[0]
            current_training_image_set = [image]
        else:
            current_training_image_set.append(image)

    image_training_values[result_of_image] = build_training_data_for_image_set(current_training_image_set)

    return  image_training_values


def build_training_data_for_image_set(training_images: []):
    image_pixel_values = np.zeros((50, 50), dtype=float)

    for image in training_images:
        for y in range(0, len(image)):
            for x in range(0, len(image[0])):
                color = image[y][x]

                # Computing the luminance. Works fine if the color space is not gamma compressed
                luminance = 0.2126 * float(color[2]) + 0.7152 * float(color[1]) + 0.0722 * float(color[0])

                if luminance > 128:  # White pixel
                    image_pixel_values[y][x] += float(1/len(training_images))

    return image_pixel_values
