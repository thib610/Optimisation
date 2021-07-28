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
    acc=[]
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
        print(evalXold)
        print(evalXbest)

        ###########################################################################fefinir ca
        if evalXbest>evalXold:
            flagimp=1
            evalXold=copy.deepcopy(evalXbest)
            Xold=copy.deepcopy(Xbest)
        elif evalXbest<evalXold:
            flagimp = 0
            combo.append(Xold)
            acc.append(evalXold)
            lemachinstop += 1
        else:
            flagimp = 0

        th = updateth(th, nactp, nimp, M)
        print(lemachinstop)
    return (combo,acc)
DIM=2#dimension du cube
NBpoint=50#nombre de point dans le cube
J=10
M=50
th=0
optilocaux=30
#LH=ESE(DIM,NBpoint,J,M,th)
LH=np.array([[0.9, 0.2],[0.5 ,0.4],[0.7, 0.8], [0.3, 0. ],[0.1, 0.6], [0.2, 0.3], [0.0 , 0.9], [0.6, 0.1], [0.8, 0.5], [0.4, 0.7]])
combo=ESE(DIM,NBpoint,J,M,optilocaux)
a=combo[0][combo[1].index(max(combo[1]))]
print(maximin(a,DIM,NBpoint))
