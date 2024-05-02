import random
import os

# alphabet is all lowercase letters in a list
alphabet = ['a','b','c','d','e']

dir = "formal_language_data"

def generate_strings_with_abc(length1, length2, n, substring='abc'):
    strings = set()  # To store unique strings
    
    while len(strings) < n:
        length = random.randint(length1, length2)
        string = ''.join(random.choices(alphabet, k=length))
        if substring in string:
            strings.add(string)
    
    return list(strings)


def generate_strings_without_abc(length1, length2, n, substring='abc'):
    strings = set()  # To store unique strings

    while len(strings) < n:
        length = random.randint(length1, length2)
        string = ''.join(random.choices(alphabet, k=length))
        if substring not in string:
            strings.add(string)

    return list(strings)

def generate_strings_with_abc_noncontiguous(length1, length2, n):
    strings = set()

    # generate unique strings and randomly insert letters a, b, c somewhere in the string
    # but they have to appear in that order
    while len(strings) < n:
        length1 = max(3, length1)
        length = random.randint(length1, length2)
        string = ''.join(random.choices(alphabet, k=length))
        indices = random.sample(range(length), 3)
        # sort the indices
        indices.sort()
        string_list = list(string)
        string_list[indices[2]] = 'c'
        string_list[indices[1]] = 'b'
        string_list[indices[0]] = 'a'
        string = ''.join(string_list)
        strings.add(string)
    return list(strings)

def generate_strings_without_abc_noncontiguous(length1, length2, n):
    strings = set()  # To store unique strings

    # make sure no strings contain a, b, c in that order
    while len(strings) < n:

        # choose a length
        length = random.randint(length1, length2)
        string = []
        has_a = False
        has_b = False
        for i in range(length-2):
            tempalphabet = alphabet.copy()
            if has_a:
                if has_b:
                    tempalphabet.remove('c')
                    next = random.choice(tempalphabet)
                    string.append(next)
                else:
                    next = random.choice(tempalphabet)
                    if next == 'b':
                        has_b = True
            else:
                next = random.choice(tempalphabet)
                string.append(next)
                if next == 'a':
                    has_a = True

        string = ''.join(string)
        strings.add(string)
    return list(strings)

pos_unique_strings = generate_strings_with_abc(0,50, 20000)
# print(unique_strings)

print("done with with contiguous")

neg_unique_strings = generate_strings_without_abc(0,50, 20000)
# print(unique_strings)

print("done with without contiguous")

train_pos_unique_strings = pos_unique_strings[:int(len(pos_unique_strings)*0.5)]
test_pos_unique_strings = pos_unique_strings[int(len(pos_unique_strings)*0.5):]
train_neg_unique_strings = neg_unique_strings[:int(len(neg_unique_strings)*0.5)]
test_neg_unique_strings = neg_unique_strings[int(len(neg_unique_strings)*0.5):]

# append more strings to test set of longer lengths
test_pos_unique_strings += generate_strings_with_abc(50, 100, 10000)
test_pos_unique_strings += generate_strings_with_abc(100, 150, 10000)
test_pos_unique_strings += generate_strings_with_abc(150, 200, 10000)

test_neg_unique_strings += generate_strings_without_abc(50, 100, 10000)
test_neg_unique_strings += generate_strings_without_abc(100, 150, 10000)
test_neg_unique_strings += generate_strings_without_abc(150, 200, 10000)

print("done with extra strings")

# shuffle the data
random.shuffle(test_pos_unique_strings)
random.shuffle(test_neg_unique_strings)

# Create the directory if it doesn't exist
os.makedirs(dir, exist_ok=True)

with open(os.path.join(dir, 'abc/train_pos.txt'), 'w') as file:
    for string in train_pos_unique_strings:
        file.write(string + '\n')
with open(os.path.join(dir, 'abc/train_neg.txt'), 'w') as file:
    for string in train_neg_unique_strings:
        file.write(string + '\n')


with open(os.path.join(dir, 'abc/test_pos.txt'), 'w') as file:
    for string in test_pos_unique_strings:
        file.write(string + '\n')
with open(os.path.join(dir, 'abc/test_neg.txt'), 'w') as file:
    for string in test_neg_unique_strings:
        file.write(string + '\n')



pos_unique_strings = generate_strings_with_abc_noncontiguous(0,50, 20000)
# print(unique_strings)

print("done with with noncontiguous")

neg_unique_strings = generate_strings_without_abc_noncontiguous(0,50, 20000)
# print(unique_strings)

print("done with without noncontiguous")

train_pos_unique_strings = pos_unique_strings[:int(len(pos_unique_strings)*0.5)]
test_pos_unique_strings = pos_unique_strings[int(len(pos_unique_strings)*0.5):]
train_neg_unique_strings = neg_unique_strings[:int(len(neg_unique_strings)*0.5)]
test_neg_unique_strings = neg_unique_strings[int(len(neg_unique_strings)*0.5):]

# append more strings to test set of longer lengths
test_pos_unique_strings += generate_strings_with_abc_noncontiguous(50, 100, 10000)
test_pos_unique_strings += generate_strings_with_abc_noncontiguous(100, 150, 10000)
test_pos_unique_strings += generate_strings_with_abc_noncontiguous(150, 200, 10000)

test_neg_unique_strings += generate_strings_without_abc_noncontiguous(50, 100, 10000)
test_neg_unique_strings += generate_strings_without_abc_noncontiguous(100, 150, 10000)
test_neg_unique_strings += generate_strings_without_abc_noncontiguous(150, 200, 10000)

print("done with extra strings")

# shuffle the data
random.shuffle(test_pos_unique_strings)
random.shuffle(test_neg_unique_strings)

with open(os.path.join(dir, 'abc_noncontiguous/train_pos.txt'), 'w') as file:
    for string in train_pos_unique_strings:
        file.write(string + '\n')
with open(os.path.join(dir, 'abc_noncontiguous/train_neg.txt'), 'w') as file:
    for string in train_neg_unique_strings:
        file.write(string + '\n')


with open(os.path.join(dir, 'abc_noncontiguous/test_pos.txt'), 'w') as file:
    for string in test_pos_unique_strings:
        file.write(string + '\n')
with open(os.path.join(dir, 'abc_noncontiguous/test_neg.txt'), 'w') as file:
    for string in test_neg_unique_strings:
        file.write(string + '\n')