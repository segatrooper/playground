#real one

import numpy as np, csv

# playground file

def create_layer(i, o):
    return np.random.random((i, o))


def interpretfile():
    fname = ['test.csv', 'train.csv']
    fname_open = []
    ifile = []
    output = []
    for i in fname:
        i = open(i)
        fname_open.append(i)
        ifile.append(csv.DictReader(i))
    for i in ifile:
        cache = dict()
        for person in i:
            person = dict(person)
            # print(person)
            cache[person['PassengerId']] = person
        output.append(cache)
    for i in fname_open:
        i.close()
    return output


def sig(x):
    return 1/(1 + np.exp(-x))

def forwardpass(i, o):
    # U HAVE ININ OUTOUT h1 h2 out weights and biases
    oof = []
    i = np.array(i)
    z = np.dot(i, h1_weights) + h1_biases
    a = np.maximum(z,0)
    oof.append(z)
    z = np.dot(a, h2_weights) + h2_biases
    a = np.maximum(z,0)
    oof.append(z)

    z = np.dot(a, out_weights) + out_biases
    a = np.maximum(z,0)
    oof.append(z)
    print(f"printing output for fowardpass: {a}")
    return a, oof

def backprop(inin, y, actual, z_values, lrate=-1):
    print(f'printing the pre-activated z_values: {z_values}')
    for z in z_values:
        z[z > 0] = 1; z[z <= 0] = 0;
    z = z_values
    global out_err, h1_err, h2_err, w1_err, w2_err, w3_err, COUNTER, cost, out_weights, h2_weights, h1_weights, h1_biases, h2_biases, out_biases
    out_err = np.dot(actual - y, z[-1])
    h2_err = np.dot(out_err, out_weights.T)
    h2_err = np.multiply(h2_err, z[-2])

    h1_err = np.dot( h2_err, h2_weights)
    h1_err = np.multiply(h1_err, z[-3])

    w3_err = out_err * np.maximum(z[-1], 0)
    w2_err = h2_err * np.maximum(z[-2],0)
    w1_err = h1_err * np.maximum(z[-3],0)
    print(f'''printing errors:
            {w1_err}:
            {w2_err}:
            {w3_err}:''')
    h1_biases -= h1_err * lrate
    h2_biases -= h2_err * lrate
    out_biases -= out_err * lrate
    h1_weights -= w1_err * lrate
    h2_weights -= w2_err * lrate
    out_weights -= w3_err * lrate
    COUNTER += 1
    print(f'''
COUNTER = {COUNTER} - cost = {cost}:
    h1_biases = {h1_biases}
    h2_biases = {h2_biases}
    out_biases = {out_biases}
    h1_weights = {h1_weights}
    h2_weights = {h2_weights}
    out_weights = {out_weights}
    y = {y}
    actual = {actual}''')
if __name__ == '__main__':
    # modeling a 11 input to 1 output with 30 nodes of 2 layers in between

    """
    ---
    11 --- 30 --- 30 --- 1
    ---
    lets set training batches to 5 ppl each

    ---
    1 x 11 --- 11 x 30 --- 30 x 30 --- 30 x 1
    ---
    first, lets test my knowledge using XOR functions

    """
    COUNTER = 0
    ININ = [[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]]
    OUTOUT = [[0],
        [1],
        [1],
        [0]]
    h1_weights = create_layer(2, 4)
    h2_weights = create_layer(4, 4)
    out_weights = create_layer(4, 1)
    h1_biases = create_layer(1, 4)
    h2_biases = create_layer(1,4)
    out_biases = create_layer(1,1)
    while True:
        if input('quit?') == 'q': break
        for i in range(len(ININ)):
            inin = ININ[i]
            y, oof = forwardpass(ININ[i], OUTOUT)
            yb = [list(o) for o in list(y)]
            cost = np.sum((np.array(OUTOUT[i]) - np.array(yb))**2) / 2
            print(f'printing out cost: {cost}')
            print(f'printing out list of outputs: {y}')
            backprop(inin, y, OUTOUT[i], oof)
    print('ended the program')
