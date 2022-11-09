import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.stats import ortho_group
import os
import time

mode = 'pic' 
# mode 'gen' will generate random data and draw pictures
# mode 'pic' will only draw pictures from data saved before

def V(state, d, m):
    sq = np.reshape(state, (d,d))
    res = 0
    for i in range(d):
        r = 0
        for j in range(d-i):
            r += sq[j, i+j]
        res += abs(r)**2
    for i in range(1,d):
        r = 0
        for j in range(d-i):
            r += sq[i+j, j]
        res += abs(r)**2
    return m*res-m

def maxEntangledState(d):
    state = np.zeros((d*d, 1))
    for i in range(d):
        state[i*d+i] = 1/np.sqrt(d)
    return state

CZ = np.eye(4)
CZ[3, 3] = -1

CIZ = np.eye(8)
CIZ[5, 5] = -1
CIZ[7, 7] = -1
CIZI = np.kron(CIZ, np.eye(2))
ICIZ = np.kron(np.eye(2), CIZ)

CIIZ = np.eye(16)
CIIZ[9, 9] = -1
CIIZ[11, 11] = -1
CIIZ[13, 13] = -1
CIIZ[15, 15] = -1
CIIZII = np.kron(CIIZ, np.eye(4))
ICIIZI = np.kron(np.eye(2), np.kron(CIIZ, np.eye(2)))
IICIIZ = np.kron(np.eye(4), CIIZ)


ns = [1,2,3]
ts = [100, 1000, 10000]
m = 2
z = 200


xx = np.zeros((len(ns), len(ts), z))
yy = np.zeros((len(ns), len(ts), z))

for ni in range(len(ns)):
    n = 2*ns[ni]
    d = 2**n
    nu = ns[ni]
    du = 2**nu
    if (mode != 'gen') and os.path.exists("data/cost_x{}.npy".format(nu)):
        xx[ni, :] = np.load("data/cost_x{}.npy".format(nu))
        yy[ni, :] = np.load("data/cost_y{}.npy".format(nu))
        continue

    veca = np.zeros((m, d, d), dtype="complex")
    vecb = np.zeros((m, d, d), dtype="complex")

    for x in range(m):
        for i in range(d):
            for j in range(d):
                veca[x, i, j] = 1 / np.sqrt(d) * np.exp(2*np.pi*1j/d*j*(i-(x+1-1/2)/m))
                vecb[x, i, j] = 1 / np.sqrt(d) * np.exp(-2*np.pi*1j/d*j*(i-(x+1)/m))

    alpha = np.zeros(d)

    for k in range(d):
        alpha[k] = 1/(2*d) * np.tan(np.pi/(2*m)) * (1/np.tan(np.pi*(k+1/(2*m))/d))

    print(alpha)

    # - 1/np.tan(np.pi*(np.floor(d/2)+1/(2*m))/d)

    M = np.zeros((d, m, d*d, d*d), dtype="complex")

    for k in range(d):
        for i in range(m):
            for a in range(d):
                Aia = np.mat(veca[i, a, :]).T @ np.mat(veca[i, a, :]).conjugate();
                Biak = np.mat(vecb[i, (a-k)%d, :]).T @ np.mat(vecb[i, (a-k)%d, :]).conjugate();
                M[k, i, :, :] += np.kron(Aia, Biak)

    N = np.zeros((d, m, d*d, d*d), dtype="complex")

    for k in range(d):
        for a in range(d):
            for i in range(m-1):
                Ai1a = np.mat(veca[i+1, a, :]).T @ np.mat(veca[i+1, a, :]).conjugate();
                Biak = np.mat(vecb[i, (a+k)%d, :]).T @ np.mat(vecb[i, (a+k)%d, :]).conjugate();
                N[k, i, :, :] += np.kron(Ai1a, Biak)
                
            A11a = np.mat(veca[0, a-1, :]).T @ np.mat(veca[0, a-1, :]).conjugate();
            Bmak = np.mat(vecb[m-1, (a+k)%d, :]).T @ np.mat(vecb[m-1, (a+k)%d, :]).conjugate();
            N[k, m-1, :, :] += np.kron(A11a, Bmak)

    I = np.eye(du)

    for zi in range(z):
        state = maxEntangledState(d)
        U1 = ortho_group.rvs(du)
        U2p = ortho_group.rvs(du)

        w, v = np.linalg.eig(U1 @ U2p.T)

        w2 = [cmath.polar(x) for x in w]
        w2 = [cmath.rect(r, p * zi / z) for (r, p) in w2]

        U2 = v @ np.diag(w2) @ v.conjugate().T @ U2p


        c1 = np.trace(U1.T @ U2) / du
        D = np.sqrt(1-abs(c1)**2)

        if nu == 1:
            U1 = CZ @ np.kron(I, U1)
            U2 = CZ @ np.kron(I, U2)
        elif nu == 2:
            U1 = CIZI @ ICIZ @ np.kron(I, U1)
            U2 = CIZI @ ICIZ @ np.kron(I, U2)
        elif nu == 3:
            U1 = CIIZII @ ICIIZI @ IICIIZ @ np.kron(I, U1)
            U2 = CIIZII @ ICIIZI @ IICIIZ @ np.kron(I, U2)

        state = np.kron(U1, U2) @ state
        
        ppM = [[np.trace(state.conjugate().T @ M[k, x] @ state) for k in range(d)] for x in range(m)]
        ppN = [[np.trace(state.conjugate().T @ N[k, x] @ state) for k in range(d)] for x in range(m)]

        for ti in range(len(ts)):
            t = ts[ti]
            res = 0

            for tt in range(t):
                w = np.random.randint(0, 2)
                if w == 0:
                    x = np.random.randint(0, m)
                    k = np.random.choice(d, 1, p = ppM[x])
                    res += alpha[k]
                else:
                    x = np.random.randint(0, m)
                    k = np.random.choice(d, 1, p = ppN[x])
                    res += alpha[k]

            res /= t / 2
            xx[ni, ti, zi] = np.sqrt(1-abs(c1*c1))
            yy[ni, ti, zi] = res

    np.save("data/cost_x{}.npy".format(ns[ni]), xx[ni,:])
    np.save("data/cost_y{}.npy".format(ns[ni]), yy[ni,:])

ax = []


plt.figure(1, dpi=1080)
ax += [plt.subplot(3,3,1)]
ax += [plt.subplot(3,3,2, sharex=ax[0], sharey=ax[0])]
ax += [plt.subplot(3,3,3, sharex=ax[0], sharey=ax[0])]
ax += [plt.subplot(3,3,4)]
ax += [plt.subplot(3,3,5, sharex=ax[3], sharey=ax[3])]
ax += [plt.subplot(3,3,6, sharex=ax[3], sharey=ax[3])]
ax += [plt.subplot(3,3,7)]
ax += [plt.subplot(3,3,8, sharex=ax[6], sharey=ax[6])]
ax += [plt.subplot(3,3,9, sharex=ax[6], sharey=ax[6])]



plt.setp(ax[0].get_xticklabels(), visible=False)
plt.setp(ax[1].get_xticklabels(), visible=False)
plt.setp(ax[2].get_xticklabels(), visible=False)
plt.setp(ax[3].get_xticklabels(), visible=False)
plt.setp(ax[4].get_xticklabels(), visible=False)
plt.setp(ax[5].get_xticklabels(), visible=False)


plt.setp(ax[1].get_yticklabels(), visible=False)
plt.setp(ax[2].get_yticklabels(), visible=False)
plt.setp(ax[4].get_yticklabels(), visible=False)
plt.setp(ax[5].get_yticklabels(), visible=False)
plt.setp(ax[7].get_yticklabels(), visible=False)
plt.setp(ax[8].get_yticklabels(), visible=False)



plt.sca(ax[3])
plt.ylabel("Estimated Bell Expression Value $I_{d,m}$", fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12})
plt.sca(ax[7])
plt.xlabel("Actual Distance $D(U_1,U_2)$", fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12})


for i in range(len(ns)):
    for j in range(len(ts)):
        d = 2**ns[i]
        plt.sca(ax[i * 3 + j])
        plt.scatter(xx[i,j,:], yy[i,j,:]*m*d-m, s=0.5**2)
        yyy = np.linspace(0, 1, 1000)
        plt.plot(np.sqrt(1-(m*d*yyy)/(m*d)), yyy*m*d-m, linewidth=0.5, color='red')
        plt.text(0, np.min(yy[i,j,:])*m*d-m, "{}+{} qubits\n{} times".format(ns[i], ns[i], ts[j]), horizontalalignment = "left", verticalalignment='bottom',
        fontdict={'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 8})

#plt.tight_layout()
plt.savefig("pic/cost.png")

plt.show()






