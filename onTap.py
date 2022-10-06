# Exercise 1: Reverse each word of a string
import collections


def reverse_words(Sentence):
    words = Sentence.split(" ")
    # interate lít and reverse each worch using ::-1
    new_word_list = [word[::-1] for word in words] 
    
    # Joining the new list of words
    res_str = " ".join(new_word_list)
    return res_str

# Exercise 2: Read text file into a variable and replace all newlines with space
def read_replace():
    with open('sample.txt', 'r') as file:
        data = file.read().replace('\n', ' ')
        print("data", data)

# Exercise 3: Remove items from a list while iterating
def remove_list_solution1():
    number_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    i = 0
    # get list's size
    n = len(number_list)
    # iterate list till i is smaller than n
    while i < n:
        # check if number is greater than 50
        if number_list[i] >30:
            del number_list[i]
            #  reduce the list size
            n = n -1
        else:
            # moce to next item
            i = i + 1

    print(number_list)

def remove_list_solution2():
    number_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(len(number_list) - 1, -1, -1):
        if number_list[i] > 30:
            del number_list[i]
    print(number_list)

# Exercise 4: Reverse Dictionary mapping
def reverse_ascii_dict():
    ascii_dict = {'A':65, 'B':66, 'C':67, 'D':68}
    # Reverse mapping
    new_dict = {value:key for key, value in ascii_dict.items()}
    print(new_dict)
reverse_ascii_dict()

# Exercise 5: Display all duplicate items from a list
def duplicates():
    sapmple_list = [10, 20, 60, 30, 20, 40, 30, 60, 70, 80]
    duplicates = []
    for item, count in collections.Counter(sapmple_list).items():
        print(item)
        print(count)
        if count > 1:
            duplicates.append(item)
    # print(duplicates)

duplicates()

# Exercise 6: Filter dictionary to contain keys present in the given list
 
def filter_dictionaery():
    # Dictionary
    d1 = {'A': 65, 'B': 66, 'C': 67, 'D': 68, 'E': 69, 'F': 70}

    # Filter dict using following keys
    l1 = ['A', 'C', 'F']
    new_dict = {key: d1[key] for key in l1}
    print(new_dict)
filter_dictionaery()