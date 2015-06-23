"""
Greville and OPIUM method for classifying Mackey-Glass from:
J. Tapson and A. van Schaik, 
"Learning the Pseudoinverse Solution to Network Weights"
Neural Networks

Used for Figure 3.

@author: andrevanschaik
"""
from pylab import *
from numpy import *
from OPIUM import *

# Simulation parameters
dt = 0.1              # time step
maxtime = 4000        # simulation stop time
alpha =1.0            # learning rate for OPIUM

# generate Mackey-Glass series
a = 0.2
b = 0.1
tau = 170
mg = ones(maxtime)
mg[0] = 0.000001
for t in range (tau,maxtime-1):
    mg[t+1] = mg[t]+((a*mg[t-tau])/(1+(pow(mg[t-tau],10)))-b*mg[t])

# Network parameters
numtaps = 4
taps = array((0,60,170,1000))
max_taps = 1000
fanout = 10
forward = 50
size_hidden = numtaps*fanout            # size of hidden layer
random_weights = random.rand(size_hidden,numtaps)-0.5 # input->hidden weights

# Greville Method

# Initialisation of matrices
M = zeros((1,size_hidden))              # hidden->output weights
x = zeros((numtaps,1))                  # current inputs to the network
h = zeros((size_hidden,1))              # hidden layer output
E = zeros(maxtime)                      # error matrix for plotting vs time
Y = zeros((1,maxtime))                  # network output vs time
P = eye(size_hidden) / size_hidden      # initialise correlation matrix inverse

for t in range(max_taps,maxtime-forward):
    x = reshape(mg[t-taps],(numtaps,1)) # input vector 
    h = tanh(dot(random_weights,x))     # hidden layer activation with sigmoid
    y = dot(M,h)                        # output value
    Y[0,t+forward] = y                  # output is saved as the predicted sample
    E[t+forward] = mg[t+forward]-y      # calculate error
    Greville(h,E[t+forward],M,P)        # basic Greville method  
# end for t

# Calculate RMS error for the last 1000 points
error_G = sqrt(mean((Y[0,maxtime-1000:maxtime]-mg[maxtime-1000:maxtime])**2))
print error_G

# Plot input, output, and error
ion()
figure(0)
plot(mg)
plot(Y[0],'r')
plot(E,'g')

savetxt('MG_Greville.txt',(mg, Y[0], E))

# OPIUM Method

# Initialisation of signal matrices
M = zeros((1,size_hidden))              # hidden->output weights
x = zeros((numtaps,1))                  # current inputs to the network
h = zeros((size_hidden,1))              # hidden layer output
E_O = zeros(maxtime)                    # error matrix for plotting vs time
Y_O = zeros((1,maxtime))                # network output vs time
P = eye(size_hidden) / size_hidden      # initialise correlation matrix inverse

for t in range(max_taps,maxtime-forward):
    x = reshape(mg[t-taps],(numtaps,1)) # input vector 
    h = tanh(dot(random_weights,x))     # hidden layer activation with sigmoid
    y = dot(M,h)                        # output value
    Y_O[0,t+forward] = y                # output is saved as the predicted sample
    E_O[t+forward] = mg[t+forward]-y    # calculate error
    OPIUM(h,E_O[t+forward],M,P,alpha)   # OPIUM method       
# end for t

# Calculate RMS error for the last 1000 points
error_O = sqrt(mean((Y_O[0,maxtime-1000:maxtime]-mg[maxtime-1000:maxtime])**2))
print error_O

# Plot input, output, and error
figure(1)
plot(mg)
plot(Y_O[0],'r')
plot(E_O,'g')

savetxt('MG_Opium.txt',(mg, Y_O[0], E_O))
