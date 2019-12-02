import numpy as np
x = np.array([[2,9], [3, 6], [4,8]])
y = np.array([[92], [86], [89]])
x = x/np.amax(x,axis=0)
y = y/100

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

epoch = 1000
eta = 0.1
input_neurons = 2
hidden_neurons = 3
output_neurons = 1
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
for i in range(epoch):
    h_ip = np.dot(x, wh)
    h_act = sigmoid(h_ip)
    o_ip = np.dot(h_act, wout)+bout
    output = sigmoid(o_ip)
    Eo = y-output
    outgrad = sigmoid_grad(output)
    d_output = Eo*outgrad
    Eh = np.dot(d_output, wout.T)
    hiddengrad = sigmoid_grad(h_act)
    d_hidden = Eh*hiddengrad
    wout += np.dot(h_act.T, d_output)*eta 
    wh += np.dot(x.T, d_hidden)*eta
print("Normalized Input: \n", x)
print("Actual Output: \n", y)
print("Predicted Output: \n", output)
