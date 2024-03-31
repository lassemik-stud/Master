import math
kt = [['11'],['12'],['13'],['14'],['15'],['16'],['17'],['18'],['19'],['20'],['21']]
ut = [['21'],['22'],['23'],['24'],['25'],['26'],['27'],['28'],['29'],['30'],['31]']]
c = [1]*len(ut)

# ROLLING ATTRIBUTION PARAMETERS
N = len(ut)
k = 4 
d = 2
n = math.ceil((N-k)/(k-d)) + 1
l = len(kt)*n 
w = k-d
l_group = len(kt)

print('N\tk\td\tn\tl\tw\tl_group')
print(f'{N}\t{k}\t{d}\t{n}\t{l}\t{w}\t{l_group}')

x_out = []
y_out = []

# ROLLING SELECTION
for j, s_kt in enumerate(kt):
    i = 0
    while i < n*w:
        flat_list = [item for sublist in ut[i:i+k] for item in sublist]
        x_out.append([s_kt, [' '.join(flat_list)]])
        print(f'i: {i} \t c: {c[i:i+k]}')
        y_out.append(0 if any(element == 0 for element in c[i:i+k]) else 1)
        i+=w

for part in x_out:
    print(part)
print(y_out)
N_theoretical = (n-1)*(k-d) + k
N_val = [[] for _ in range(N_theoretical)]

count_c = 0
for elements in range(int(l/n)):
    j = 0
    for element in range(n):
        for i in range(k):
            try:
                print(f'elements: {elements} element: {element} of {n}\t j: {j} - {ut[j]} - C: {y_out[count_c]}')
            except:
                pass
            N_val[j].append(y_out[count_c])
            j+=1
        print('\n')
        count_c+=1
        j-=d

print(N_val)
for i, part in enumerate(N_val[:N]):
    array_sum = sum(part)
    count_zero = part.count(0)
    count_ones = part.count(1)
    result = array_sum if array_sum == 0 else array_sum / len(part)
    print(f'{ut[i]} {part} \t-> {result} -\t Weight: {(count_ones-count_zero)/len(kt)}')


