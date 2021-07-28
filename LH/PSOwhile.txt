import sys

print(sys.version)
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy


def initgrid(NBpoint, DIM):
    grid = np.zeros((DIM, NBpoint), dtype=float)
    for i in range(0, DIM):
        for j in range(0, NBpoint):
            grid[i, j] = j
    return grid


# choisi un nombre aléatoire dans un tableau
def RandInArray(A):
    a = A[np.random.randint(0, np.size(A))]
    return a


# génére un point aléatoirement a partir de la grille
def GenPoint(grid, DIM):
    res = np.zeros(DIM, dtype=float)
    j = 0
    for i in grid:
        res[j] = RandInArray(i)
        j = j + 1
    return res


# enléve un point donné dans la grille
def delpointingrid(grid, point):
    j = 0
    (a, b) = np.shape(grid)
    res = np.zeros((a, b - 1), dtype=float)
    for i in point:
        res[j] = np.delete(grid[j], np.where(grid[j] == point[j]))
        j = j + 1
    return res


# génére un LH avec le systeme expliquer précement
def getLH(DIM, NBpoint):
    grid = initgrid(NBpoint, DIM)
    LH = np.zeros((NBpoint, DIM), dtype=float)
    for i in range(0, NBpoint):
        p = GenPoint(grid, DIM)
        LH[i] = p
        grid = delpointingrid(grid, p)
    for i in range(0, DIM):
        for j in range(0, NBpoint):
            LH[j, i] = LH[j, i] / NBpoint
    return LH


################################################################################################
#############################PARTIE II Algorithme génétic#######################################
################################################################################################
#################1) le principe
# https://www.youtube.com/watch?v=1i8muvzZkPw heu c'est mieux que du texte pour expliquer
#################2) les fonctions
# distance euclidienne en N dimmension
def d(x, y, DIM):
    acc = 0
    for i in range(0, DIM):
        acc = acc + (x[i] - y[i]) * (x[i] - y[i])
    acc = math.sqrt(acc)
    return acc


# critére d'évaluation maximin
def maximin(LH, DIM, NBpoint):
    acc = np.zeros(NBpoint, dtype=float)
    accfinal = np.zeros(NBpoint, dtype=float)
    for i in range(0, NBpoint):
        for j in range(0, NBpoint):
            a = d(LH[i], LH[j], DIM)
            if a == 0:
                acc[j] = 999990
            else:
                acc[j] = a  # trouver un moyen que cette case sois pas le minimum
        accfinal[i] = min(acc)
    return min(accfinal)


def maximinn(LH, DIM, NBpoint):
    acc = 0
    for i in range(0, NBpoint - 2):
        for j in range(i + 1, NBpoint - 1):
            acc = acc + 1 / (d(LH[i], LH[j], DIM) * d(LH[i], LH[j], DIM))
    return 1 / acc


def plotLH2D(LH):
    for i in LH:
        [a, b] = i
        plt.plot([a], [b], 'ro')
        plt.axis([0, 1, 0, 1])

    return


def plotLH3D(LH):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in LH:
        [a, b, c] = i
    ax.scatter3D(LH[:, 0], LH[:, 1], LH[:, 2]);

    return


def plot(LH, DIM):
    if DIM == 2:
        plotLH2D(LH)
    else:
        plotLH3D(LH)


def move(LHH1, LHH2, speed):
    LH1 = copy.deepcopy(LHH1)
    LH2 = copy.deepcopy(LHH2)

    for j in range(0, int(random.random() * speed * 2)):
        i = j % DIM
        pos = random.randint(0, NBpoint - 1)
        acc = LH1[pos, i]
        target = LH2[pos, i]
        ind = list(LH1[:, i]).index(target)
        LH1[pos, i] = target
        LH1[ind, i] = acc
    return LH1


def randswap(LH, DIM, probR):
    for i in range(0, DIM):
        if random.uniform(0, 1) < probR:
            pos2 = random.randint(0, NBpoint - 1)
            pos1 = random.randint(0, NBpoint - 1)
            acc = LH[pos1, i]
            LH[pos1, i] = LH[pos2, i]
            LH[pos2, i] = acc
    return LH


def FindpersonalBest(PersonalBesttab, LHlist, particuleID):
    return (LHlist[particuleID])[(PersonalBesttab[particuleID]).index(max(PersonalBesttab[particuleID]))]


def FindglobalBest(PersonalBesttab, LHlist):
    lign = len(PersonalBesttab[-1])
    acc = []
    for i in PersonalBesttab:
        acc.append(i[lign - 1])
    index = acc.index(max(acc))
    LH = (LHlist[index])[lign - 1]
    return LH


def FindglobalBest2(PersonalBesttab, LHlist, NBinitialLH, i):
    acc = 0
    ii = 0
    jj = 0
    for i in range(0, NBinitialLH):
        for j in range(0, i - 1):
            if acc < PersonalBesttab[i][j]:
                acc = PersonalBesttab[i][j]
                LH = LHlist[i][j]
    return LH


def Hamdistance(M1, M2, DIM, NBpoint):
    acc = 0
    for i in range(0, NBpoint):
        for j in range(0, DIM):
            if M1[i, j] != M2[i, j]:
                acc += 1
    return acc


def PSO(DIM, NBpoint, probR, NBinitialLH):
    LHlist = [[]]
    PersonalBesttab = [[]]
    for i in range(0, NBinitialLH):
        LHlist.append([])
        PersonalBesttab.append([])
    for i in range(0, NBinitialLH):
        LH = getLH(DIM, NBpoint)
        LHlist[i].append(LH)
        PersonalBesttab[i].append(maximin(LH, DIM, NBpoint))
    LHlist.remove(LHlist[-1])
    PersonalBesttab.remove(PersonalBesttab[-1])
    maxi = max(PersonalBesttab)[0]
    acc = 0
    i = 0
    while acc < 70:
        listmax = []
        for j in range(0, NBinitialLH):
            pb = FindpersonalBest(PersonalBesttab, LHlist, j)
            gb = FindglobalBest(PersonalBesttab, LHlist)
            LH = move(LHlist[j][-1], pb, Hamdistance(pb, LHlist[j][-1], DIM, NBpoint))
            LH = move(LH, gb, Hamdistance(gb, LH, DIM, NBpoint))
            LH = randswap(LH, DIM, probR)
            LHlist[j].append(LH)
            PersonalBesttab[j].append(maximin(LH, DIM, NBpoint))
            listmax.append(max(PersonalBesttab[j]))
        i += 1
        if maxi == max(listmax):
            acc += 1
        else:
            maxi = max(listmax)
            acc = 0
        print(maxi)
        #plt.axis([0, i, 0, 1])
        #plt.scatter(i, maxi)
        #plt.pause(0.00001)
        #print(maximinn(LH, DIM, NBpoint))
        #print(i)
    #plt.show()
    return FindglobalBest2(PersonalBesttab, LHlist, NBinitialLH, i)


DIM = 2  # dimension du cube
NBpoint = 50  # nombre de point dans le cube
probR = 0.4
NBinitialLH = 100
from statistics import mean

acc = []
acc2 = []
for i in range(0, 3):
    LH = PSO(DIM, NBpoint, probR, NBinitialLH)
    acc.append(maximin(LH, DIM, NBpoint))
    acc2.append(maximinn(LH, DIM, NBpoint))
    #plot(LH, DIM)
    #plt.show()
    print(max(acc))
    print(mean(acc))
    print(max(acc2))
    print(mean(acc2))
    print("##################")
    #print(LH)
plot(LH,DIM)
