import particle_class
import globals
import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


# definition of mathematical function
def parabole(x, y):
    return (x * x + y * y)/200


def plan(x, y):
    return x * x - y * y


def himmelblau(x, y):
    return (pow((x * x + y - 11), 2) + pow((x + y * y - 7), 2))/20000


def distance(x, y):
    return math.sqrt(pow((x[0] - y[0]), 2) + pow((x[1] - y[1]), 2))


# algorithm code
def pso(fct, affichage):
    # particle initialisation
    global_best = [100, 100, 99999999999]
    particles = []
    for i in range(0, globals.NB_PARTICLE):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        pb = [x, y, fct(x, y)]

        particles.append(particle_class.Particle(x, y, pb, [0, 0]))
        if global_best[2] > pb[2]:
            global_best[0] = x
            global_best[1] = y
            global_best[2] = pb[2]
    # update of particles for each iteration
    #
    history = []
    for i in range(0, globals.MAXITER):
        if affichage and i % 1 == 0:
            history.append(copy.deepcopy(particles))
        for j in range(0, globals.NB_PARTICLE):
            particles[j].update_particle(fct, global_best)
    return global_best, history


# plot function


def show_plot(etats, fct):
    # definition of axis
    plt.ion()
    x = np.linspace(-10, 10, 30)
    y = np.linspace(-10, 10, 30)
    x_axis, y_axis = np.meshgrid(x, y)
    z_axis = fct(x_axis, y_axis)
    plt.figure()
    # plot everything at every step
    for swarm in etats:
        ax = plt.axes(projection='3d')
        ax.view_init(60, 25)
        ax.plot_surface(x_axis, y_axis, z_axis, alpha=0.8, rstride=1, cstride=1, cmap="Reds", edgecolor='none')
        for particle in swarm:
            plt.plot(particle.get_pos()[0],
                     particle.get_pos()[1],
                     fct(particle.get_pos()[0], particle.get_pos()[1]),
                     'ro')
        plt.pause(0.1)
        plt.clf()


if __name__ == '__main__':
    # run pso
    global_best, states = pso(himmelblau, globals.AFFICHAGE)
    print(global_best)
    # printing particle swarm
    if globals.AFFICHAGE:
        show_plot(states, himmelblau)
