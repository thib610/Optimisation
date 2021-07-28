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
    print(LH)
    for i in range(0,DIM):
        for j in range(0,NBpoint):
            LH[j,i]=LH[j,i]/NBpoint
    return  LH
################################################################################################
#############################PARTIE II Algorithme génétic#######################################
################################################################################################
#################1) le principe
#https://www.youtube.com/watch?v=1i8muvzZkPw heu c'est mieux que du texte pour expliquer
#################2) les fonctions
#distance euclidienne en N dimmension
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
def maximinn(LH,DIM,NBpoint):
    acc=0
    for i in range(0,NBpoint-2):
        for j in range(i+1,NBpoint-1):
            acc=acc+1/(d(LH[i], LH[j], DIM)*d(LH[i], LH[j], DIM))
    return 1/acc

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
DIM=2
NBpoint=50
LH=getLH(DIM,NBpoint)
print(LH)
print(maximin(LH,DIM,NBpoint))
print(maximinn(LH,DIM,NBpoint))
print("################")
a=np.zeros((50,2))
a=np.array([[0	,13],[1	,27],[2	,41],[3	,5],[4	,19],[5	,33],[6	,47],[7	,11],[8	,25],[9,39],[10,3],[11,17],[12,31],[13,45],[14,9],[15,23],[16,37],[17,1],[18,15],[19,29],[20,43],[21,7],[22,21],[23,	35],[24,	49],[25,	0],[26,	14],[27,	28],[28,	42],[29,	6],[30,	20],[31,	34],[32,	48],[33,	12],[34,	26],[35,	40],[36,	4],[37,	18],[38,	32],[39,	46],[40,	10],[41,	24],[42,	38],[43,	2],[44,	16],[45,	30],[46,	44],[47,	8],[48,	22],[49,	36]])
a=a/50
print(maximin(a,DIM,NBpoint))
print(maximinn(a,DIM,NBpoint))
print("################")
b=np.array([[0,	2],[1,	5],[2,	8],[3,	1],[4,	4],[5,	7],[6,	0],[7,	3],[8,	6],[9,	9]])
b=b/10
NBpoint=10
print(maximin(b,DIM,NBpoint))
print(maximinn(b,DIM,NBpoint))
print("################")
b=np.array([[0	,9	,12],
[1	,18	,25],
[2	,27	,38],
[3	,36	,0 ],
[4	,45	,13],
[5	,4	,26],
[6	,13	,39],
[7	,22	,1 ],
[8	,31	,14],
[9	,40	,27],
[10	,49	,40],
[11	,8	,2 ],
[12	,17	,15],
[13	,26	,28],
[14	,35	,41],
[15	,44	,3 ],
[16	,3	,16],
[17	,12	,29],
[18	,21	,42],
[19	,30	,4 ],
[20	,39	,17],
[21	,48	,30],
[22	,7	,43],
[23	,16	,5 ],
[24	,25	,18],
[25	,34	,31],
[26	,43	,44],
[27	,2	,6 ],
[28	,11	,19],
[29	,20	,32],
[30	,29	,45],
[31	,38	,7 ],
[32	,47	,20],
[33	,6	,33],
[34	,15	,46],
[35	,24	,8 ],
[36	,33	,21],
[37	,42	,34],
[38	,1	,47],
[39	,10	,9 ],
[40	,19	,22],
[41	,28	,35],
[42	,37	,48],
[43	,46	,10],
[44	,5	,23],
[45	,14	,36],
[46	,23	,49],
[47	,32	,11],
[48	,41	,24],
[49	,0	,37]])
b=b/49
NBpoint=50
DIM=3
print(maximin(b,DIM,NBpoint))
print(maximinn(b,DIM,NBpoint))


print("################")
b=np.array([[0	,14	,23	,33	,41],
[1	,41	,13	,32	,30],
[2	,17	,30	,21	,2 ],
[3	,27	,24	,4	,32],
[4	,18	,1	,19	,28],
[5	,25	,32	,45	,18],
[6	,35	,8	,7	,8 ],
[7	,31	,48	,20	,22],
[8	,3	,14	,37	,16],
[9	,45	,29	,27	,5 ],
[10	,42	,35	,23	,46],
[11	,28	,3	,35	,6 ],
[12	,6	,47	,36	,29],
[13	,15	,42	,14	,47],
[14	,4	,41	,8	,19],
[15	,0	,18	,11	,38],
[16	,34	,7	,16	,48],
[17	,24	,6	,47	,34],
[18	,7	,12	,9	,9 ],
[19	,26	,36	,46	,45],
[20	,47	,43	,41	,26],
[21	,30	,33	,1	,10],
[22	,49	,19	,13	,25],
[23	,44	,17	,48	,15],
[24	,22	,25	,25	,27],
[25	,1	,22	,43	,39],
[26	,5	,34	,34	,3 ],
[27	,32	,45	,40	,1 ],
[28	,46	,16	,38	,43],
[29	,21	,10	,0	,31],
[30	,36	,40	,5	,37],
[31	,10	,0	,26	,35],
[32	,29	,21	,24	,0],
[33	,48	,44	,18	,12],
[34	,13	,9	,44	,11],
[35	,38	,4	,3	,7],
[36	,19	,38	,49	,23],
[37	,37	,2	,29	,21],
[38	,11	,46	,30	,44],
[39	,8	,37	,2	,33],
[40	,20	,49	,22	,17],
[41	,9	,31	,6	,4],
[42	,16	,20	,17	,49],
[43	,39	,39	,31	,40],
[44	,43	,11	,12	,42],
[45	,2	,27	,28	,24],
[46	,12	,5	,15	,13],
[47	,23	,15	,42	,36],
[48	,40	,28	,39	,14],
[49	,33	,26	,10	,20]])
b=b/49
NBpoint=50
DIM=5
print(maximin(b,DIM,NBpoint))
print(maximinn(b,DIM,NBpoint))

print("################")
b=np.array([[0	,14	,13	,49	,37	,18	,33	,40	,23	,33],
[1	,34	,7	,8	,32	,26	,39	,21	,3	,14],
[2	,25	,48	,37	,14	,23	,41	,7	,27	,40],
[3	,3	,31	,12	,8	,29	,15	,9	,34	,9],
[4	,21	,39	,17	,19	,47	,42	,47	,28	,21],
[5	,49	,22	,44	,20	,32	,28	,23	,32	,2],
[6	,36	,5	,30	,10	,21	,7	,2	,21	,36],
[7	,41	,27	,15	,22	,20	,17	,36	,49	,48],
[8	,9	,24	,10	,9	,19	,22	,35	,5	,49],
[9	,42	,45	,26	,27	,2	,25	,45	,10	,22],
[10	,30	,46	,1	,38	,35	,11	,10	,16	,35],
[11	,24	,15	,22	,6	,0	,48	,29	,39	,18],
[12	,17	,26	,24	,44	,37	,3	,42	,12	,5],
[13	,20	,29	,42	,40	,46	,9	,11	,47	,31],
[14	,22	,40	,46	,2	,30	,1	,38	,24	,29],
[15	,8	,16	,6	,34	,36	,46	,12	,40	,41],
[16	,31	,42	,16	,49	,16	,37	,24	,48	,10],
[17	,11	,6	,38	,1	,42	,45	,17	,15	,23],
[18	,45	,25	,39	,30	,45	,26	,30	,1	,43],
[19	,38	,17	,0	,25	,5	,2	,22	,29	,6],
[20	,12	,0	,33	,39	,17	,29	,4	,37	,4],
[21	,6	,1	,21	,16	,38	,10	,43	,43	,27],
[22	,23	,43	,41	,33	,10	,5	,3	,13	,11],
[23	,39	,19	,31	,46	,1	,36	,6	,22	,42],
[24	,1	,41	,13	,31	,8	,43	,19	,7	,15],
[25	,7	,8	,34	,12	,3	,12	,34	,6	,8],
[26	,2	,12	,32	,42	,33	,16	,8	,2	,38],
[27	,35	,38	,14	,0	,31	,21	,18	,0	,7],
[28	,16	,10	,4	,45	,7	,30	,48	,26	,32],
[29	,0	,23	,36	,18	,4	,18	,13	,42	,44],
[30	,48	,4	,23	,3	,22	,23	,49	,18	,26],
[31	,44	,2	,11	,48	,40	,14	,20	,30	,30],
[32	,27	,35	,28	,36	,48	,44	,0	,17	,13],
[33	,4	,37	,47	,21	,28	,34	,33	,38	,3],
[34	,13	,49	,35	,47	,25	,27	,37	,25	,46],
[35	,32	,18	,43	,35	,6	,4	,41	,45	,19],
[36	,46	,36	,2	,23	,24	,49	,27	,19	,39],
[37	,29	,14	,3	,15	,39	,38	,28	,41	,0],
[38	,15	,44	,7	,5	,11	,20	,44	,36	,24],
[39	,19	,32	,18	,4	,49	,19	,14	,31	,47],
[40	,43	,34	,27	,11	,15	,24	,1	,46	,20],
[41	,47	,47	,25	,29	,44	,13	,39	,33	,17],
[42	,33	,11	,40	,28	,34	,47	,32	,44	,37],
[43	,26	,33	,48	,7	,9	,40	,25	,8	,34],
[44	,18	,3	,5	,13	,13	,31	,5	,14	,28],
[45	,40	,20	,29	,43	,14	,35	,31	,11	,1],
[46	,37	,28	,20	,26	,12	,0	,26	,9	,45],
[47	,28	,9	,45	,17	,41	,6	,16	,20	,12],
[48	,10	,21	,19	,24	,43	,32	,46	,4	,25],
[49	,5	,30	,9	,41	,27	,8	,15	,35	,16]])
b=b/49
NBpoint=50
DIM=10
print(maximin(b,DIM,NBpoint))
print(maximinn(b,DIM,NBpoint))


print("################")
b=np.array([[0,	4],
[1,	9],
[2,	14],
[3,	19],
[4,	1],
[5,	6],
[6,	11],
[7,	16],
[8,	3],
[9,	8],
[10,	13],
[11,	18],
[12,	0],
[13,	5],
[14,	10],
[15,	15],
[16,	2],
[17,	7],
[18,	12],
[19,	17]])
b=b/19
NBpoint=20
DIM=2
print(maximin(b,DIM,NBpoint))
print(maximinn(b,DIM,NBpoint))