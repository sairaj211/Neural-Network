import math
import matplotlib.pyplot as plt
import random

of_weights = []
af_weights = []

# Activation Function
def tanh(x):
    return math.tanh(x)

def d_tanh(x):
    return ((1/math.cosh(x))**2)

# Activation Function
def sigmoid(x):
    return (1/(1+math.exp(-x)))

def d_sigmoid(x):
    return (math.exp(-x)/(math.exp(-x)+1)**2)

#########################################################################

Training_data = 30   

# Number of neurons 
neurons = 300        

Training_Inputs = []
Training_Outputs = []

#Input(x) to layer 1 weights
L1_weights = []

#Layer 1 to Output(y) weights
L2_weights = []

#########################################################################

#Generate Randoms weights
for i in range(0,neurons):
    L1_weights.append(random.random())
    L2_weights.append(random.random())

#NN learning rate
Learn_Rate = 0.1

#Traning Inputs and Outputs 
for i in range(1,Training_data+1):
    x_i = (i - 1)/Training_data
    Training_Inputs.append(x_i)
    Training_Outputs.append(x_i*x_i)
    
Total_error = 1
Iter_count = 0
Max_iter = 2000

print("Approximation of X^2 function")

#########################################################################

#Evaulated output
e_outs = []

#Calculate error and update each neuron 
while (Total_error > 0.40 and Iter_count < Max_iter ):
    
    Total_error = 0.0
    e_outs = []
    
    for k in range(0,Training_data):
       
        #Feed Forward
        o0 = tanh(Training_Inputs[k])
        z = []
        oi = []
        
        for i in range(0,neurons):
            z.append(o0 * L1_weights[i])
            oi.append(tanh(z[i]))
            
        zo = 0.0
        for i in range(0,neurons):
            zo += oi[i] * L2_weights[i]
            
        y = tanh(zo)
        e_outs.append(y)
        error = (y - Training_Outputs[k])
        Total_error += math.fabs(error)
            
        #Back propogation
        dE_don = error
        
        dE_do = []
        for i in range(0,neurons):
            dE_do.append(dE_don * d_tanh(zo) * L2_weights[i])
        
        dE_do0 = 0.0
        for i in range(0,neurons):
            dE_do0 += dE_do[i] * d_tanh(z[i]) * L1_weights[i]
            
        dE_dai = []
        dE_dwi = []
        
        for i in range(0,neurons):
            dE_dai.append(dE_do[i] * d_tanh(z[i]) * o0)
            dE_dwi.append(dE_don * d_tanh(zo) * oi[i])
            
        for i in range(0,neurons):
            L1_weights[i] = L1_weights[i] - Learn_Rate * dE_dai[i]
            L2_weights[i] = L2_weights[i] - Learn_Rate * dE_dwi[i]
            
    
    Iter_count += 1

#########################################################################
  
print("Error in approximating function X^2: ",Total_error)

#Plot the results
plot_x = Training_Inputs
plot_y = Training_Outputs
plot_NN = e_outs

plt.plot(plot_x, plot_y , label = "Expected values")
plt.plot(plot_x, plot_NN , label = "NN output")


plt.legend()

plt.show()

#########################################################################    