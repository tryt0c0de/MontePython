import numpy as np
import matplotlib.pyplot as plt
import random


def wavefunction1(x, alpha):
    return np.exp(-alpha * x ** 2)


def wavefunction2(x1, x2, y1, y2, lam, alpha):
    squaredsum = x1**2+x2**2+y1**2+y2**2
    distance = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return np.exp(-squaredsum/2) * np.exp((lam*distance)/(1+alpha*distance))


f1 = wavefunction1
f2 = wavefunction2


# I defined the other variables as standard None since I don't want to use them for the 1D case.
def manhattan(function, alpha, lam, dim, stepsize=1, n=100000):
    if dim == 1:
        x1_old = 0
        z_points = []
        for a in range(n):
            u = random.uniform(0, 1)
            z_new = random.uniform(x1_old - stepsize, x1_old + stepsize)
            p = (function(z_new, alpha)) ** 2 / (function(x1_old, alpha)) ** 2
            if p > 1:
                x1_old = z_new
            elif p > u:
                x1_old = z_new
            if a > 300:
                z_points.append(x1_old)
        return np.array(z_points)
    else:
        x1_old = 1
        x2_old = 0
        y1_old = 1
        y2_old = 0
        points = [[],[],[],[]]
        for a in range(n):
            lista1 = [x1_old, x2_old, y1_old, y2_old]
            for index, value in enumerate(lista1):
                lista2 = lista1.copy()
                u = random.uniform(0, 1)
                z_new = random.uniform(value - stepsize, value + stepsize)
                lista2[index] = z_new
                p = function(lista2[0], lista2[1], lista2[2], lista2[3], lam, alpha)**2/function(lista1[0], lista1[1], lista1[2], lista1[3], lam, alpha)**2
                if p > 1:
                    lista1[index] = z_new
                elif p > u:
                    lista1[index] = z_new
                if a > 300:
                    points[index].append(lista1[index])
        return np.array(points)


def energy(z, alpha):
    e_l = alpha + z ** 2 * (1 / 2 - 2 * alpha ** 2)
    average = np.mean(e_l)
    sigma = np.std(e_l)
    return [average, sigma]


def derivative(xfirst, xsecond, lam , alpha, r):
    der =-1+ lam/((1+alpha*r)**2*r)-lam*((xfirst-xsecond)**2)/((1+alpha*r)**3*(r**3))*(1+3*alpha*r)+(-xfirst+lam*(xfirst-xsecond)/((1+alpha*r)**2*r))**2
    return der


def energy2(x1,x2,y1,y2,alpha,lam):
    r = np.sqrt((x1-x2)**2+(y1-y2)**2)
    dx1 = derivative(x1,x2,lam, alpha,r)
    dx2 = derivative(x2,x1,lam, alpha,r)
    dy1 = derivative(y1, y2,lam, alpha, r)
    dy2 = derivative(y2, y1,lam, alpha, r)
    term1 = -0.5*(dx1+dy1+dx2+dy2) + 0.5*(x1**2+x2**2 +y1**2 + y2**2)
    en = term1-lam/r
    average = np.mean(en)
    sigma = np.std(en)
    return [average, sigma]


def plot1():
    alphas = np.linspace(0.1, 1, 100)
    energies = []
    std = []
    for a in alphas:
        z = manhattan(f1, a,lam = None, dim = 1)
        energies.append(energy(z, a)[0])
        std.append(energy(z,a)[1])
    energies = np.array(energies)
    std = np.array(std)
    plt.grid()
    plt.title("One Particle")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Energy")
    plt.plot(alphas, energies, label = "Average Energy")
    plt.plot(alphas,std, label = "Standard Deviation")
    plt.legend()
    plt.show()
    print(np.min(energies))


def plot_hist(alpha):
    x = np.linspace(-3,3)
    z = manhattan(f1, alpha, lam = None, dim = 1)
    gauss = 1/np.sqrt(np.pi)*np.exp(-x*x)
    plt.grid()
    plt.title(r"$\psi_{1}$")
    plt.plot(x,gauss)
    plt.hist(z, bins=30, density = True)
    plt.show()


def truegold(a, b, threshold, f):
    gr = (np.sqrt(5)-1)/2
    while (b-a) > threshold:
        d = gr * (b - a)
        x_1 = a+d
        x_2 = b-d
        z_1 = manhattan(f, x_1, dim = 1,lam = None)
        e_1 = energy(z_1, x_1)[0]
        z_2 = manhattan(f, x_2, dim = 1, lam = None)
        e_2 = energy(z_2, x_2)[0]
        if e_1 < e_2:
            a = x_2
        else:
            b = x_1
    print("The minimum alpha is found at:\n ", "alpha = ", x_1)
    print("------------------------------")
    print("With average energy:", energy(z_1, x_1)[0], "And with a standard deviation of: ", energy(z_1, x_1)[1])
    return a, b, x_1, energy(z_1, x_1)


def gold2(a, b, threshold, f, lam):
    gr = (np.sqrt(5)-1)/2
    while (b-a) > threshold:
        d = gr * (b - a)
        x_1 = a+d
        x_2 = b-d
        z_1 = manhattan(f, x_1, lam, dim = 2)
        e_1 = energy2(z_1[0],z_1[1],z_1[2], z_1[3], x_1, lam)[0]
        z_2 = manhattan(f, x_2,lam,  dim = 2)
        e_2 = energy2(z_2[0],z_2[1],z_2[2], z_2[3], x_2, lam)[0]
        if e_1 < e_2:
            a = x_2
        else:
            b = x_1
    print("The minimum alpha is found at:\n ", "alpha = ", x_1)
    print("------------------------------")
    print("With average energy:", energy2(z_1[0],z_1[1],z_1[2], z_1[3], x_1, lam)[0], "And with a standard deviation of: ", energy2(z_1[0],z_1[1],z_1[2], z_1[3], x_1, lam)[1])
    return a, b, x_1, energy(z_1, x_1)


def plot2(lam):
    alphas = np.linspace(0.1, 1, 100)
    energies2 = []
    for a in alphas:
        z2 = manhattan(f2, a, lam,  dim = 2)
        energies2.append(energy2(z2[0],z2[1],z2[2], z2[3], a, lam)[0])
    energies = np.array(energies2)
    plt.grid()
    plt.title(f"Two particles with lamda ={lam}")
    plt.xlabel("Alpha")
    plt.ylabel("Energy")
    plt.plot(alphas, energies2)
    plt.show()
    print(np.min(energies2))
    
#plot1()
#plot2(0, 20)
plot2(1)
#plot2(2)
#plot2(4)
#plot2(6)
#plot1()
# plot_hist(1/2)
#plot1()
#print(gold2(0.1, 1.5, 10e-7,f2,0))
print(gold2(0.1, 1.5, 10e-7,f2,1))
# print(min_energy, min_alpha)
#print(energy2(1,3,0,0,1,1))
# plot_hist(0.5)
