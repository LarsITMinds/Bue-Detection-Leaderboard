from PIL import Image
import cv2
import os
import screen_detection
import text_detection
import numpy as np

from Classes.Player import Player
from training import get_training_data
from util import sort_list_of_symbols, find_most_matching_word, group_words_by_player

get_letters_for_training = False
create_pictures = True
training_set = {}


def main():
    training_images = load_images_from_folder('Test data/Letters & numbers')

    global training_set
    training_set = get_training_data(training_images)

    images = load_images_from_folder('Test data/Test data')
    #screen_detection.find_scoreboard_screen(images)

    adapted_images = load_images_from_folder('Test data/Screen detection results')

    for index in range(0, len(adapted_images)):
        if adapted_images[index][1] == "26":
            print("Finding words for image: " + adapted_images[index][1])
            predicted_texts = predict_scoreboard_text(adapted_images[index][0], adapted_images[index][1])

            print(predicted_texts)

            player_words = group_words_by_player(predicted_texts, adapted_images[index][0].shape[1])

            sorted_player_words = []
            print(" ")
            print(player_words)
            for i in range(len(player_words)):
                #print("Stats for player", i + 1)
                sorted_player_words.append([])
                for predicted_word_info in player_words[i]:
                    break_outer = False
                    predicted_word = predicted_word_info[0]
                    most_matched_word, distance = find_most_matching_word(predicted_word)

                    for j in range(len(sorted_player_words[i])):
                        player_stat = sorted_player_words[i][j]
                        if player_stat[0] == most_matched_word:
                            if player_stat[1] > distance:
                                sorted_player_words[i][j] = [most_matched_word, distance, predicted_word]
                                break_outer = True
                                break
                            else:
                                break

                    if break_outer:
                        break

                    sorted_player_words[i].append([most_matched_word, distance, predicted_word])

                    #print("Predicted word: " + predicted_word + " - Most matched word: " + most_matched_word)
                #print(" ")

            print(" ")
            print(sorted_player_words)

            players = []
            for i in range(len(sorted_player_words)):
                players.append(Player(i + 1))
                for word_info in sorted_player_words[i]:
                    if word_info[0] == "Kills":
                        amount = word_info[2].split(" ")[1]
                        players[i].kills = amount
                    elif word_info[0] == "Deaths":
                        amount = word_info[2].split(" ")[1]
                        players[i].deaths = amount
                    elif word_info[0] == "Self":
                        amount = word_info[2].split(" ")[1]
                        players[i].self = amount
                    else:
                        players[i].awards.append(word_info[0])

            for player in players:
                player.print_information()


def predict_scoreboard_text(original_image: Image, file_name: str):
    print("Predicting Texts")
    image = original_image.copy()
    possible_test_contours = text_detection.find_texts_on_screen(image, file_name, create_pictures)

    rect_image = image.copy()

    predicted_words_sections = []

    for index in range(0, len(possible_test_contours)):
        x = possible_test_contours[index][0]
        y = possible_test_contours[index][1]
        width = possible_test_contours[index][2]
        height = possible_test_contours[index][3]
        cv2.rectangle(rect_image, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 2)



        #text_areas.append(image[y: y + height, x: x + width])
        #if file_name == "24":
        print("Predicting possible word: " + str(index) + " out of " + str(len(possible_test_contours)))
        image_of_word = image[y: y + height, x: x + width]
        predicted_word = predict_word(image_of_word, file_name + str(index))

        if len(predicted_word) > 0:
            ##### Finding whitespace in symbol collections #####

            #print(predicted_word)

            total_word_length = predicted_word[0][-1][0][1] - predicted_word[0][0][0][1]
            total_symbols_length = 0

            for i in range(len(predicted_word[0])):
                if i + 1 >= len(predicted_word[0]):
                    break
                # print(sorted_symbol_collection[0][index][0][2])
                total_symbols_length = total_symbols_length + predicted_word[0][i][0][2]

            #print(total_symbols_length)
            #print((total_word_length - total_symbols_length) / (len(predicted_word[0]) - 1))

            average_distance_between_symbols = (total_word_length - total_symbols_length) / (
                        len(predicted_word[0]) - 1)
            average_distance_factor = 2
            found_whitespaces = 0

            for j in range(len(predicted_word[0])):
                base_index = j + found_whitespaces
                compare_index = j + found_whitespaces + 1
                if compare_index >= len(predicted_word[0]):
                    break

                symbol_x_end_pos = predicted_word[0][base_index][0][1] + predicted_word[0][base_index][0][2]
                compare_symbol_x_start_pos = predicted_word[0][compare_index][0][1]

                #print(symbol_x_end_pos)
                #print(compare_symbol_x_start_pos)
                #print()

                if compare_symbol_x_start_pos - symbol_x_end_pos > average_distance_between_symbols * average_distance_factor:
                    #print("Possible space here between: " + predicted_word[0][base_index][1] + " and " +
                    #      predicted_word[0][compare_index][1])
                    predicted_word[0].insert(compare_index, [[], ' '])
                    found_whitespaces = found_whitespaces + 1

            ####################################################
            #print(predicted_word)
            predicted_words_sections.append([predicted_word, possible_test_contours[index]])

    predicted_words = text_detection.look_for_and_combine_possible_words(predicted_words_sections, image.shape[0], image.shape[1])

    if create_pictures:
        cv2.imwrite('Score results/Images/Text area images/' + file_name + ' text areas.jpg', rect_image)
        #cv2.imwrite('Score results/Images/' + file_name + ' text areas.jpg', text_area)

    return predicted_words


def predict_word(original_image, file_name):
    image = original_image.copy()
    text_detection.remove_brownish_color(image, file_name, False)
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    (_, bw) = cv2.threshold(gray_image, 120, 255.0, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #print(len(contours))

    mask = np.zeros(bw.shape, dtype=np.uint8)

    symbol_collection = []

    for index in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[index])

        mask[y:y + h, x:x + w] = 0
        cv2.drawContours(mask, contours, index, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

        if float(w/h) > 1:
            ratio_aspect = float(h/w)
        else:
            ratio_aspect = float(w/h)

        if 0.50 < r < 0.92 and ratio_aspect > 0.55 and h > 20:
            cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
            symbol_area = bw[y: y + h, x: x + w]
            if get_letters_for_training:
                cv2.imwrite('Test data/Letters & numbers/Unsorted training data/' + str(file_name) + ' ' + str(index) + '.jpg', symbol_area)
            else:
                predicted_symbol = find_matching_symbol(cv2.resize(symbol_area, (50, 50)))
                symbol_collection.append(((y, x, w, h), predicted_symbol))

            #cv2.imshow('Compressed image', cv2.resize(symbol_area, (50, 50)))
            #cv2.imshow('Original image', symbol_area)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    #symbol_collection.sort(key=lambda e: e[0][1])
    #print(symbol_collection)

    sorted_symbol_collection = sort_list_of_symbols(symbol_collection)

    #tempWord = ''
    #for symbol_group in sorted_symbol_collection:
    #    for symbol in symbol_group:
    #        tempWord += symbol[1]
    #    tempWord += ' '
    #print(tempWord)


    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(bw, kernel, iterations=1)

    #cv2.imshow('Original image', image)
    #cv2.imshow('BW image', bw)
    #cv2.imshow('erosion image', erosion)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return sorted_symbol_collection


def find_matching_symbol(symbol_to_match):
    symbol_value_dictionary = {}

    for symbol in training_set.keys():
        symbol_value_dictionary[symbol] = 0
        for y in range(0, len(symbol_to_match)):
            for x in range(0, len(symbol_to_match[0])):
                pixel_value = training_set[symbol][y][x]
                color = symbol_to_match[y][x]

                if color > 128:  # White pixel
                    symbol_value_dictionary[symbol] += float(pixel_value)
                else:  # Black
                    symbol_value_dictionary[symbol] += 1 - float(pixel_value)

    current_possible_symbol = ''
    current_value = 0
    for symbol in symbol_value_dictionary.keys():
        if current_value < symbol_value_dictionary[symbol]:
            current_value = symbol_value_dictionary[symbol]
            current_possible_symbol = symbol
        #print(symbol, symbol_value_dictionary[symbol])

    #print('Symbol might be: ' + current_possible_symbol + ' with score: ' + str(current_value))


    return current_possible_symbol


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append([img, filename.split('.jpg')[0]])
    return images

main()
