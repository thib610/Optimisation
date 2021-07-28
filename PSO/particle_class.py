import globals
import random


class Particle:
    def __init__(self, x, y, pb, direction):
        self.x = x
        self.y = y
        self.pb = pb
        self.direction = direction

    def get_pos(self):
        return self.x, self.y

    def update_pos(self, x, y):
        self.x = x
        self.y = y

    def get_pb(self):
        return self.pb

    def update_pb(self, pb):
        self.pb = pb

    def get_direction(self):
        return self.direction

    def update_direction(self, direction):
        self.direction = direction

    def update_particle(self, fct, global_best):
        new_direction = [globals.INERTIA * self.direction[0] +
                         globals.COGNITIVE * random.uniform(0, 1) * (self.pb[0] - self.x) +
                         globals.SOCIAL * random.uniform(0, 1) * (global_best[0] - self.x),

                         globals.INERTIA * self.direction[1] +
                         globals.COGNITIVE * random.uniform(0, 1) * (self.pb[1] - self.y) +
                         globals.SOCIAL * random.uniform(0, 1) * (global_best[1] - self.y)]

        self.x += new_direction[0] * globals.LR
        self.y += new_direction[1] * globals.LR
        self.direction = new_direction
        tmp = fct(self.x, self.y)
        if self.pb[2] > tmp:
            self.pb[2] = tmp
            self.pb[1] = self.y
            self.pb[0] = self.x
            if self.pb[2] < global_best[2]:
                global_best[0] = self.x
                global_best[1] = self.y
                global_best[2] = self.pb[2]
