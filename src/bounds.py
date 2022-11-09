import numpy as np
import matplotlib.pyplot as plt
import cmath
import os

from scipy.stats import ortho_group

mode = 'pic'
# mode 'gen' will generate random data and draw pictures
# mode 'pic' will only draw pictures from data saved before

ns = [2,3,4]

z = 10000
m = 2

X = np.zeros((2,2))
X[0, 1] = 1
X[1, 0] = 1
Z = np.zeros((2,2))
Z[0, 0] = 1
Z[1, 1] = -1
CZ = np.eye(4)
CZ[3, 3] = -1
H = np.zeros((2,2))
H[0, 0] = 1
H[1, 0] = 1
H[0, 1] = 1
H[1, 1] = -1
H /= np.sqrt(2)
I = np.eye(2)

xx = np.zeros((len(ns), z))
yy = np.zeros((len(ns), z))


for ni in range(len(ns)):

    if (mode != 'gen') and os.path.exists("data/bounds_x{}.npy".format(ns[ni])):
        xx[ni, :] = np.load("data/bounds_x{}.npy".format(ns[ni]))
        yy[ni, :] = np.load("data/bounds_y{}.npy".format(ns[ni]))
        continue

    d = 2**ns[ni]


    veca = np.zeros((d, d, m), dtype="complex")
    vecb = np.zeros((d, d, m), dtype="complex")

    for x in range(m):
        for i in range(d):
            for j in range(d):
                veca[i, j, x] = 1 / np.sqrt(d) * np.exp(2*np.pi*1j/d*j*(i-(x+1-1/2)/m))
                vecb[i, j, x] = 1 / np.sqrt(d) * np.exp(-2*np.pi*1j/d*j*(i-(x+1)/m))


    rhoa = np.zeros((d, d, d, m), dtype="complex")
    rhob = np.zeros((d, d, d, m), dtype="complex")
    for i in range(m):
        for a in range(d):
                rhoa[a, :, :, i] = np.mat(veca[a, :, i]).T @ np.mat(veca[a, :, i]).conjugate();
                rhob[a, :, :, i] = np.mat(vecb[a, :, i]).T @ np.mat(vecb[a, :, i]).conjugate();

    alpha = np.zeros(d)

    for k in range(d):
        alpha[k] = 1/(2*d) * np.tan(np.pi/(2*m)) * (1/np.tan(np.pi*(k+1/(2*m))/d))

    #print(alpha)

    # - 1/np.tan(np.pi*(np.floor(d/2)+1/(2*m))/d)

    M = np.zeros((d*d, d*d, d), dtype="complex")

    for k in range(d):
        for i in range(m):
            for a in range(d):
                Aia = rhoa[a, :, :, i]
                Biak = rhob[(a-k)%d, :, :, i]
                M[:, :, k] += np.kron(Aia, Biak)

    #N = np.zeros((d, m, d*d, d*d), dtype="complex")

    for k in range(d):
        for a in range(d):
            for i in range(m-1):
                Ai1a = rhoa[a, :, :, i+1]
                Biak = rhob[(a+k)%d, :, :, i]
                M[:, :, k] += np.kron(Ai1a, Biak)
                
            A11a = rhoa[a-1, :, :, 0]
            Bmak = rhob[(a+k)%d, :, :, m-1]
            M[:, :, k] += np.kron(A11a, Bmak)

    M = alpha * M
    #print(alpha * M)
    #print(M)

    B = np.sum(M, axis=2)


    #B = np.sum([alpha[k] * M[k] for k in range(d)])

    Psi = np.zeros((d * d, 1), dtype="complex")

    for i in range(d):
        Psi[i * d + i] = 1 / np.sqrt(d)

    for k in range(z):

        U1 = ortho_group.rvs(d)
        U2p = ortho_group.rvs(d)

        w, v = np.linalg.eig(U1 @ U2p.T)

        w2 = [cmath.polar(x) for x in w]
        w2 = [cmath.rect(r, p * k / z) for (r, p) in w2]

        U2 = v @ np.diag(w2) @ v.conjugate().T @ U2p

        c1 = np.trace(U1.T @ U2) / d
        #perp = (np.random.randn(d*d, 1) * 2 - 1) + (np.random.randn(d*d, 1) * 2 - 1) * 1j

        perp = np.kron(U1, U2) @ Psi

        #perp = perp - (Psi.T @ perp) * Psi

        perp = perp / np.linalg.norm(perp)

        #print(np.linalg.norm(perp))

        rhop = perp @ perp.conjugate().T

        res1 = np.trace(rhop @ B)
        res1 /= m
        #print(V(perp, d, m), m*d*res1-m)
        #print(t, np.real(res1))
        xx[ni, k] = np.sqrt(1-abs(c1*c1))
        yy[ni, k] = res1

    
    np.save("data/bounds_x{}.npy".format(ns[ni]), xx[ni,:])
    np.save("data/bounds_y{}.npy".format(ns[ni]), yy[ni,:])

    ax = []

    plt.figure(1, dpi=1000)




    plt.ylabel("V", fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12})
    plt.xlabel("D(U1,U2)", fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12})

    yyy = np.linspace(0, 1, 1000)


    #plt.hist(vs[:,i], 100, density=True)
    F = 0.98
    plt.scatter(xx[ni,:], yy[ni,:]*m*d-m, s=0.5**2)
    cu = np.sqrt((m*d*yyy)/(m*d))
    cl = np.sqrt((m*d*yyy-m-m*(d-2))/m)
    plt.plot(np.sqrt(1-cu*cu), yyy*m*d-m, linewidth=1, color='orange', label='lower bound')
    plt.plot(np.sqrt(1-cl*cl), yyy*m*d-m, linewidth=1, color='red', label='upper bound')

    plt.savefig("pic/bounds{}.png".format(ns[ni]))

    plt.close()
