from Data.data import possible_words
from Levenshtein import distance as levenshtein_distance


def sort_list_of_symbols(symbol_collection):
    # Sorting by X coordinate
    symbol_collection.sort(key=lambda e: e[0][1])

    current_symbol_group = []

    for symbol_data in symbol_collection:
        if not check_for_existing_group(symbol_data, current_symbol_group):
            #print('Found no existing group')
            current_symbol_group.append([symbol_data])

    # Removing groups under the length of 3. There is no group of words with length 1 or 2... I think :-)
    filtered_symbol_group = [e for e in current_symbol_group if len(e) >= 3]
    # Sorting by Y coordinate of the last entry in each group.
    filtered_symbol_group.sort(key=lambda e: e[0][0])

    #print(filtered_symbol_group)
    return filtered_symbol_group


def check_for_existing_group(symbol_data, current_symbol_group) -> bool:
    for index in range(0, len(current_symbol_group)):
        symbol_group = current_symbol_group[index]
        last_symbol_y = symbol_group[-1][0][0]
        last_symbol_height = symbol_group[-1][0][3]

        if last_symbol_y - 5 < symbol_data[0][0] < last_symbol_y + 5 or \
                last_symbol_y + last_symbol_height - 5 < symbol_data[0][0] + symbol_data[0][3] < last_symbol_y + last_symbol_height + 5:
            #print("Found existing group")
            current_symbol_group[index].append(symbol_data)
            return True

    return False

def group_words_by_player(predicted_texts: [str, [int, int, int, int]], image_width):
    player_one = []
    player_two = []
    player_three = []
    player_four = []

    average_player_border = image_width / 4
    #print("width: " + str(average_player_border))
    for predicted_word_info in predicted_texts:
        predicted_word_position = predicted_word_info[1]

        #print("x position: " + str(predicted_word_position[0]))
        #print(predicted_word_position[0] > average_player_border * 3)

        if predicted_word_position[0] > average_player_border * 3:
            #print("Adding " + predicted_word_info[0] + " to player 4")
            player_four.append(predicted_word_info)
        elif predicted_word_position[0] > average_player_border * 2:
            #print("Adding " + predicted_word_info[0] + " to player 3")
            player_three.append(predicted_word_info)
        elif predicted_word_position[0] > average_player_border:
            #print("Adding " + predicted_word_info[0] + " to player 2")
            player_two.append(predicted_word_info)
        else:
            #print("Adding " + predicted_word_info[0] + " to player 1")
            player_one.append(predicted_word_info)

    return [player_one, player_two, player_three, player_four]


def find_most_matching_word(predicted_word):
    guessed_word = ""
    current_distance = 42
    for word in possible_words:
        word_distance = levenshtein_distance(predicted_word.lower(), word.lower())
        if word_distance < current_distance:
            guessed_word = word
            current_distance = word_distance

    #print("Distance: ", current_distance)

    return guessed_word
