kt = [1,2,3,4,5,6,7,8,9,10]

k = 3                       # window size
N = len(kt)                 # total length
d = 2                       # overlap size 
n = int((N-k)/(k-d)) + 1    # number of windows

elements = []

print(kt)

for i in range(N):
    window_content = kt[i:i+k]
    if len(window_content) == k:
        elements.append(kt[i:i+k])
        i += d
    else: 
        break

print(elements)

# Initialize a new list with zeros
reversed_kt = [0]*N

# For each window
for j in range(len(elements)):
    # For each position in the window
    for i in range(len(elements[j])):
        # Add the window's value to the corresponding position in reversed_kt
        reversed_kt[i+j] += elements[j][i]

print(reversed_kt)