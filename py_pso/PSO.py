import numpy as np
from pyrsistent import *
from adamantine import *



class PSO:
    def __init__(self, n_particles, n_dimensions, bounds, w, c1, c2, max_iter, verbose=False):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.verbose = verbose

        self.particles = pvector(self._initialize_particles())
        
        self.best_global_position = self.particles[0][0]
        self.best_global_value = float('inf')

    def _velocity(self, particle):
        pos, vel, best_pos, best_val = particle 
        inertia = self.w * vel
        cognitive = self.c1 * np.random.rand() * (best_pos - pos)
        social = self.c2 * np.random.rand() * (self.best_global_position - pos)
        return inertia + cognitive + social
    

    # clip x to be within bounds
    def _trim(self, x, bounds):
        if x < bounds[0] or x > bounds[1]:
            return np.random.uniform(bounds[0], bounds[1])
        else:
            return x

    

    def _clip_position(self, position):
        
        pos = pvector(parmap(self._trim , position, self.bounds))
        return pos
    

    def _position(self, particle):
        pos, vel, best_pos, best_val = particle
        npos = pos + vel

        npos = self._clip_position(npos)
        return np.array(npos)    

    def iteration(self, objective_function):
        
        def update_particle(particle):
            pos, vel, best_pos, best_val = particle
            nvel = self._velocity(particle)
            npos = self._position((pos, nvel, best_pos, best_val))
            value = objective_function(npos)
            if value < best_val:
                best_val = value
                best_pos = npos
            return (npos, nvel, best_pos, best_val)

        
        self.particles = pvector(parmap(update_particle, self.particles))

        self.best_global_position, self.best_global_value = \
            foldl(lambda acc, x: x[3] < acc[1] and (x[2], x[3]) or acc, (self.best_global_position, self.best_global_value), self.particles)

        return self.best_global_position, self.best_global_value

    def _initialize_particles(self):
        for i in range(self.n_particles):
            position = np.array(pvector(map(lambda x: np.random.uniform(x[0], x[1]), self.bounds)))
            #position = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dimensions)
            velocity = np.random.uniform(-1, 1, self.n_dimensions)
            yield (position, velocity, position, float('inf'))



if __name__ == '__main__':
    def objective_function(x):
        return np.sum(x**2)

    pso = PSO(10, 2, ((-10, 10), (-10, 10)), 0.7, 1.5, 1.5, 1000)
    for i in range(100):
        pso.iteration(objective_function)
        print(pso.best_global_position, pso.best_global_value)  