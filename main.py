# This is the testing for neutral network. I will create the kaggle file here
import random
import csv
def T(weight):
    output = []
    for x in range(len(weight[0])):
        output.append([])
    print(output)
    for y in range(len(weight)):
         for x in range(len(weight[0])):
            output[y].append(weight[x][y])
    return output


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

class Node:
    def __init__(self, nodes: list, layers):
        self.w = []
        
        for a in range(layers):
            self.w.append([list() for i in range(nodes[a])])
        print('done making layers')
        # each layer will have a specified number of nodes
        
        # Thus, each node will have an implited number of inputs depending on
        # the number of inputs of the previous layer of nodes
        # Ex;

        # layer

        # node

        # weight
        print(f'''self.w looks like this:
        {self.w}''')
        for i, a in enumerate(self.w[1:]):
            print(f'starting with layer {i + 1}')
            print(f'    looks like this: {a}')
            for c, b in enumerate(a):
                b.append([random.randint(0,1) for c in range(len(self.w[i]))]) # This is fw prop weights
                b.append(random.randint(0,1)) # This is fw prop biases
                b.append(0) # This is fw prop z values
                b.append(0) # This is fw prop a values
                b.append(0) # This is bw prop dE/db value
                b.append([0 for c in range(len(self.w[i]))]) # This is bw dE/dw values
                print(f'working with layer {i + 1} on node {c} that now has {len(b[0])} weights with bias {b[1]}')
        print('done initialization')
    def forward(self, l:list):
        '''l will be the list of inputs to to into the node:
            Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
            they will be foward propagated into a single output'''
        for i, v in enumerate(self.w[1:]):
            # we are now at the first hidden layer
            if i == 0:
                for node in v:
                    # node[2] = node[1]
                    for weightnum, weight in node[0]:
                        node[2] += (wieght * l[weightnum])
                    node[3] = relu(node[2], node[1])
                    node[4] = 0
                    node[5] = [0 for c in range(len(v))]
            else:
                for node in v:
                    # node[2] = node[1]
                    for weightnum, weight in node[0]:
                        node[2] += (weight * self.w[i][weightnum][2])
                    node[3] = relu(node[2], node[1])
        print([node[2] for node in self.w[-1]])
        print('done with the foward pass')
    def backward(self, ans):
        error = [(ans[n] - self.w[-1][n][2])**2 / 2 for n in range(len(ans))]
        Cost = []
        for i, e in enumerate(error):
            # dE/dw = dE/da * da/dz * dz/dw

            # first part: dE/da
            # E = (ans - a)**2 /2
            # dE/da = ans - a

            # second part: da/dz
            # a = ReLu(dz)
            # da/dz = d(ReLu)(dz)

            # third part: dz/dw
            # z = w * a(L-1) + C + V
            # dz/dw = a(L-1)

            # All together
            # Cost.insert(0, [ans - self.w[-1][i][2],\
             #        (ans - self.w[-1][i][2])*\
             #        (sum([self.w[-2][weightnum][2] for weightnum, weight in self.w[-1][i][0]])\
             #        if sum([weight ] else 0)*\
             #        (self.w[-2][j][2])]) 
            i = len(self.w) - 1
            for k in range(len(self.w[i])):
                for j in range(len(self.w[i-1])):
                    self.w[i][k][5][j] += (ans[k] - self.w[i][k][3]) * (1 if self.w[i][k][2] > self.w[i][k][1] else 0) * (self.w[j][k][3])
                self.w[i][k][4] += (ans[k] - self.w[i][k][3]) * (1 if self.w[i][k][2] > self.w[i][k][1] else 0) # This is for bprop biases
            for i in range(len(self.w) - 1, 1, -1):
                for k in range(len(self.w[i])):
                    self.w[i-1][k][4] += self.w[i][j][0][k] * self.w[i][k][4] * (1 if self.w[i-1][k][2] > self.w[i-1][k][1] else 0)
                    for j in range(len(self.w[i])):
                        self.w[i-1][k][5][j] += self.w[i][j][0][k] * self.w[i][k][4] * (1 if self.w[i-1][k][2] > self.w[i-1][k][1] else 0) * self.w[i-1][k][1]

            i = 1
            for k in range(len(self.w[i])):
                self.w[i-1][k][4] += self.w[i][j][0][k] * self.w[i][k][4] * (1 if self.w[i-1][k][2] > self.w[i-1][k][1] else 0)
                for j in range(len(self.w[i])):
                    self.w[i-1][k][5][j] += self.w[i][j][0][k] * self.w[i][k][4] * (1 if self.w[i-1][k][2] > self.w[i-1][k][1] else 0) * self.w[i-1][k][1]


if __name__ == '__main__':
    output = interpretfile()
    for o in output:
        print(o)
    for i in range(3):
        print('_'*20)
        print('because there are these many attributes:Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked: there will be 11 input nodes')
    test = Node([11, 20, 20, 1], 4)
    # for i in range(4):
        # print(test.w[i])

    for i in range(3):
        print('_'*20)
    print('testing transposing matrix')
    test_input = [[1,2,3],
                [4,5,6],
                [7,8,9]]
    test_output = [[1,4,7],
                [2,5,8],
                [3,6,9]]
    print(f'transposing test_input and test_output holds to be {T(test_input) == test_output}')
    print('TIME TO GET INPUTS')

