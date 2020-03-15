import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

n_cases = np.asarray([2, 8, 13, 23, 50, 109, 169, 200, 239, 267, 314, 399, 559, 689]) # Source https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Belgium
R0 = 4 # Parameter to tune so that the model fits n_cases data

sigma = 1/5.2 # Source: https://www.nature.com/articles/s41421-020-0148-0
gamma = 1/18.0 # Source: idem
beta = R0*gamma
N = 11.4*10**6 # Belgian population
I0 = 2 # Two cases on March 1st
E0 = 40*I0 # In general, E0 is 30 to 40 times higher than I0
S0 = N-I0-E0
z0 = [S0,E0, I0]
ts = 0.0
tf = 25.0
Dt = 1.0
t  = np.arange(ts, tf+Dt, Dt)
n_cases = np.asarray([2, 8, 13, 23, 50, 109, 169, 200, 239, 267, 314, 399, 559, 689]) # Source https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Belgium

# Model definition
def model(z,t,beta,sigma,gamma):
	S = z[0]
	E = z[1]
	I = z[2]
	dSdt = - beta*S*I/N
	dEdt = beta*S*I/N - sigma*E
	dIdt = sigma*E - gamma*I
	return [dSdt, dEdt, dIdt]

# Model solve
z = odeint(model,z0,t,args=(beta,sigma,gamma))
S = z[:,0]
E = z[:,1]
I = z[:,2]
R = N - (S+E+I)

#Ploting
plt.figure(figsize=(20,10))
plt.subplot(411)
plt.plot(t,S, '-g', label='Susceptibles')
plt.ylabel('Susceptibles')
plt.subplot(412)
plt.plot(t,E, '-m', label='Exposed')
plt.ylabel('Exposed')
plt.subplot(413)
plt.plot(t,I, '-r', label='Infectious')
plt.scatter([i for i in range(14)], n_cases, facecolor='red')
plt.ylabel('Infectious')
plt.ylabel('Infected')
plt.subplot(414)
plt.plot(t,R, '-k', label='Recovereds')
plt.xlabel('Time (Days)')
plt.ylabel('Recovereds')
plt.savefig('./SEIR_time.png')
plt.show()
