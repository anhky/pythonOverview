# Exercise 1: Reverse each word of a string
def reverse_words(Sentence):
    words = Sentence.split(" ")
    # interate lít and reverse each worch using ::-1
    new_word_list = [word[::-1] for word in words] 
    
    # Joining the new list of words
    res_str = " ".join(new_word_list)
    return res_str
str1 = "Dao Anh Ky"
print(reverse_words(str1))

# Exercise 2: Read text file into a variable and replace all newlines with space
def read_replace():
    with open('sample.txt', 'r') as file:
        data = file.read().replace('\n', ' ')
        print("data", data)
print(read_replace())

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
remove_list_solution1()

def remove_list_solution2():
    number_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(len(number_list) - 1, -1, -1):
        if number_list[i] > 30:
            del number_list[i]
    print(number_list)
remove_list_solution2()


