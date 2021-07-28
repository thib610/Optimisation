import sys
print(sys.version)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import random
import operator
from statistics import mean
import copy

def initgrid(NBpoint,DIM):
    grid=np.zeros((DIM,NBpoint),dtype=float)
    for i in range(0,DIM):
        for j in range(0,NBpoint):
            grid[i,j]=j
    return grid
#choisi un nombre aléatoire dans un tableau
def RandInArray(A):
    a=A[np.random.randint(0, np.size(A))]
    return a
#génére un point aléatoirement a partir de la grille
def GenPoint(grid,DIM):
    res=np.zeros(DIM,dtype=float)
    j=0
    for i in grid:
        res[j]=RandInArray(i)
        j=j+1
    return res
#enléve un point donné dans la grille
def delpointingrid(grid,point):
    j=0
    (a,b)=np.shape(grid)
    res=np.zeros((a,b-1),dtype=float)
    for i in point:
        res[j]=np.delete(grid[j],np.where(grid[j] ==  point[j]))
        j=j+1
    return res
#génére un LH avec le systeme expliquer précement
def getLH(DIM,NBpoint):
    grid = initgrid(NBpoint, DIM)
    LH=np.zeros((NBpoint,DIM),dtype=float)
    for i in range(0,NBpoint):
        p = GenPoint(grid, DIM)
        LH[i]=p
        grid=delpointingrid(grid,p)
    for i in range(0,DIM):
        for j in range(0,NBpoint):
            LH[j,i]=LH[j,i]/NBpoint
    return  LH
def d(x,y,DIM):
    acc=0
    for i in range(0,DIM):
        acc=acc+(x[i]-y[i])*(x[i]-y[i])
    acc=math.sqrt(acc)
    return acc
#critére d'évaluation maximin
def maximin(LH,DIM,NBpoint):
    acc=np.zeros(NBpoint,dtype=float)
    accfinal=np.zeros(NBpoint,dtype=float)
    for i  in range(0,NBpoint):
        for j in range(0,NBpoint):
            a=d(LH[i], LH[j], DIM)
            if a==0:
                acc[j]=999990
            else:
                acc[j]=a#trouver un moyen que cette case sois pas le minimum
        accfinal[i]=min(acc)
    return min(accfinal)
def plotLH2D(LH):
    for i in LH:
        [a,b]=i
        plt.plot([a],[b],'ro')
        plt.axis([0, 1, 0, 1])

    return
def plotLH3D(LH):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in LH:
        [a,b,c]=i
    ax.scatter3D(LH[:,0],LH[:,1],LH[:,2]);

    return
def plot(LH,DIM):
    if DIM == 2:
        plotLH2D(LH)
    else:
        plotLH3D(LH)
def randswap(LH,i):
    pos2 = random.randint(0, NBpoint - 1)
    pos1 = random.randint(0, NBpoint - 1)
    acc = LH[pos1, i]
    LH[pos1, i] = LH[pos2, i]
    LH[pos2, i] = acc
    return LH
def inneloop(X,nactp,nimp,J,M,th):
    LHlist=[]
    LHlisteval=[]
    Xbest=copy.deepcopy(X)
    eval = maximin(Xbest, DIM, NBpoint)
    for i in range(0,J):
        LHlist.append(X)
        LHlisteval.append(eval)
    for i in range(0,M):
        for j in range(0,J):
            ii=i%DIM
            LHlist[j]=randswap(LHlist[j],ii)
            LHlisteval[j] = maximin(LHlist[j], DIM, NBpoint)
        indexXtry=LHlisteval.index(max(LHlisteval))
        Xtry=LHlist[indexXtry]
        eval=maximin(Xbest,DIM,NBpoint)
        if LHlisteval[indexXtry]>eval:
            Xbest=copy.deepcopy(Xtry)
            eval=LHlisteval[indexXtry]
            nimp+=1
        elif eval-LHlisteval[indexXtry]> th*random.random():
            X=Xtry
            nactp+=1
    return (Xbest,nactp,nimp)
def updateth(th,nactp,nimp,M):
    ratioaccpt=nactp/M
    ratiopimp=nimp/M
    alpha=0.8
    if ratioaccpt>0.1 and ratiopimp<ratioaccpt:
        th=th*alpha
    #if ratioaccpt>0.1 and ratioaccpt-0.01<ratiopimp<ratioaccpt+0.01
    elif ratioaccpt > 0.1 and  ratiopimp == ratioaccpt:
        pass
    else:
        th=th/alpha
    return th

def ESE(DIM,NBpoint,J,M,optilocaux):
    combo=[]
    X=getLH(DIM,NBpoint)
    Xbest=X
    evalXbest=maximin(Xbest, DIM, NBpoint)
    th=0.005*maximin(Xbest,DIM,NBpoint)
    lemachinstop=0
    Xold = Xbest
    while lemachinstop < optilocaux:
        i=0
        nactp=0
        nimp=0
        (Xbest,nactp,nimp)=inneloop(Xbest,nactp,nimp,J,M,th)
        evalXbest=maximin(Xbest,DIM,NBpoint)
        evalXold=maximin(Xold,DIM,NBpoint)
        print(evalXbest)
        ###########################################################################fefinir ca
        if evalXbest>evalXold:
            flagimp=1
        else:
            flagimp=0
            combo.append(Xold)
            lemachinstop += 1
        th=updateth(th,nactp,nimp,M)

        print(lemachinstop)
    return combo
DIM=2#dimension du cube
NBpoint=10#nombre de point dans le cube
J=10
M=50
th=0
optilocaux=30
#LH=ESE(DIM,NBpoint,J,M,th)
LH=np.array([[0.9, 0.2],[0.5 ,0.4],[0.7, 0.8], [0.3, 0. ],[0.1, 0.6], [0.2, 0.3], [0.0 , 0.9], [0.6, 0.1], [0.8, 0.5], [0.4, 0.7]])
combo=ESE(DIM,NBpoint,J,M,optilocaux)
print(type(combo))
print(type(combo[0]))

def move(LHH1,LHH2):
    LH1=copy.deepcopy(LHH1)
    LH2=copy.deepcopy(LHH2)
    for i in range(0,DIM):
        pos = random.randint(0, NBpoint-1)
        acc=LH1[pos,i]
        target=LH2[pos, i]
        ind = list(LH1[:, i]).index(target)
        LH1[pos, i]=target
        LH1[ind, i] = acc
    return LH1
def randswap(LH,DIM,probR):
    for i in range(0,DIM):
        if random.uniform(0, 1) < probR:
            pos2 = random.randint(0, NBpoint - 1)
            pos1 = random.randint(0, NBpoint - 1)
            acc = LH[pos1, i]
            LH[pos1, i] = LH[pos2, i]
            LH[pos2, i] = acc
    return LH
def FindpersonalBest(PersonalBesttab,LHlist,particuleID):
    return (LHlist[particuleID])[(PersonalBesttab[particuleID]).index(max(PersonalBesttab[particuleID]))]
def FindglobalBest(PersonalBesttab, LHlist):
    lign=len(PersonalBesttab[-1])
    acc=[]
    for i in PersonalBesttab:
        acc.append(i[lign-1])
    index=acc.index(max(acc))
    LH=(LHlist[index])[lign-1]
    return LH
def FindglobalBest2(PersonalBesttab, LHlist,NBinitialLH,itePSO):
    acc=0
    ii=0
    jj=0
    for i in range(0,NBinitialLH):
        for j in range(0,itePSO-1):
            if acc<PersonalBesttab[i][j]:
                acc=PersonalBesttab[i][j]
                LH = LHlist[i][j]
    return LH
def PSO(DIM,NBpoint,probR,NBinitialLH,itePSO,initlist):
    LHlist = [[]]
    PersonalBesttab = [[]]
    for i in range(0,NBinitialLH):
        LHlist.append([])
        PersonalBesttab.append([])
    for i in range(0,NBinitialLH):
        LHlist[i].append(initlist[i])
        PersonalBesttab[i].append(maximin(initlist[i], DIM, NBpoint))
    LHlist.remove(LHlist[-1])
    print(type(LHlist))
    print(type(LHlist[0][0]))
    PersonalBesttab.remove(PersonalBesttab[-1])
    for i in range(0,itePSO-1):
        listmax=[]
        for j in range(0, NBinitialLH):
            LH=move(LHlist[j][-1],FindpersonalBest(PersonalBesttab,LHlist,j))
            LH=move(LHlist[j][-1], FindpersonalBest(PersonalBesttab, LHlist, j))
            LH=move(LH,FindglobalBest(PersonalBesttab,LHlist))
            LH=randswap(LH,DIM,probR)
            LHlist[j].append(LH)
            PersonalBesttab[j].append(maximin(LH, DIM, NBpoint))
            listmax.append(max(PersonalBesttab[j]))
        maxi=max(listmax)
        print(maxi)
        #plt.axis([0, i,0, 1])
        #plt.scatter(i, maxi)
        #plt.pause(0.00001)
        print(i)
    plt.show()
    return FindglobalBest2(PersonalBesttab, LHlist,NBinitialLH,itePSO)
DIM=2 #dimension du cube
NBpoint=10#nombre de point dans le cube
probR=0.2
NBinitialLH=optilocaux
itePSO=300
from statistics import mean
acc=[]
for i in range(0,1):
    LH = PSO(DIM, NBpoint, probR, NBinitialLH,itePSO,combo)
    acc.append(maximin(LH,DIM,NBpoint))
    print(i)
    print(max(acc))
    print(mean(acc))
    plot(LH,DIM)
    print(LH)
    plt.show()
    print("##################")

