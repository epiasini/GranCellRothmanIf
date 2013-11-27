# playing around with some ideas from Moreno-Bote and Parga 2006.
import numpy as np
from scipy.integrate import quad, dblquad
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

mu = 100.
sigma = 2
theta = -40
h = -65
tau_s = 30.
tau_m = 6.

z_0 = -5
t_lims = (1, 100)
z_lims = (-10, 10)
t_range = np.logspace(-0.5, 2, num=30)
z_range = np.linspace(z_lims[0], z_lims[1], 100)

gamma = np.sqrt(tau_m/tau_s)
theta_hat = np.sqrt(2)*(theta - mu*tau_m)
h_hat = np.sqrt(2)*(h - mu*tau_m)/(sigma*np.sqrt(tau_m))


def p(z, t, z_0):
    # equation 8
    return np.exp(-np.square(z-z_0*np.exp(-t/tau_s)) / (2*(1-np.exp(-2*t/tau_s)))) / np.sqrt(2*np.pi*(1 - np.exp(-2*t/tau_s)))

def j(z, t, z_0):
    # equation 6
    return (-theta_hat + gamma * z) * p(z, t, z_0) / tau_m

def j_lim(z):
    # j_lim = lim_{t->infinity} j(z,t,z_0)
    return j(z, t_lims[1], z_0=(z_lims[0]+z_lims[1])/2)

nu = quad(j_lim, -np.inf, +np.inf)[0]

def c_integrand(z_0, z, t):
    return j_lim(z_0) * j(z, t, z_0) / nu

def c(t):
    # equation 7
    # remember that in dblquad y is the first argument
    return dblquad(c_integrand,
                   -np.inf,
                   +np.inf,
                   lambda x: -np.inf,
                   lambda x: +np.inf,
                   args=(t,))

fig, ax = plt.subplots()

#Z, T = np.meshgrid(z_range, t_range)
#P = p(Z, T, z_0=z_0)
#J = j(Z, T, z_0=z_0)

#ax[0].imshow(P.transpose(), cmap='coolwarm', origin='lower', aspect='auto', interpolation='none')
#ax[1].imshow(J.transpose(), cmap='coolwarm', origin='lower', aspect='auto', interpolation='none')
c_values = []
for t in t_range:
    print("integrating for {}".format(t))
    c_values.append(c(t)[0])
print c_values
c_values = np.array(c_values)
ax.plot(t_range, c_values, color='k', linewidth=1.5)

plt.show()
