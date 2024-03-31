first_col_values = [(4,4), (3,3)]
second_col_values = [1000, 2000]
third_col_value = 'tfidf'
fourth_col_value = 'char'
fifth_col_values = [30, 40, 50]
sixth_col_value = 100
seventh_col_values = [4, 5]
eighth_col_values = [1, 2, 3, 4]

args = [
    (first, second, third_col_value, fourth_col_value, fifth, sixth_col_value, seventh, eighth)
    for first in first_col_values
    for second in second_col_values
    for fifth in fifth_col_values
    for seventh in seventh_col_values
    for eighth in eighth_col_values
]

print(len(args))