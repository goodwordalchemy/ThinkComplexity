from Life import LifeViewer, Life
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class SandPile(Life):
    
    def __init__(self, k=4, n=10):
        self.n = n
        self.k = k
        self.array = np.zeros((n,n), dtype=np.int)
        a = np.full((n-2,n-2), k, dtype=np.int)
        self.array[1:n-1,1:n-1] = a
        
    def neighbors(self, row, col):
        if row == 0 and col==0:
            return [(row+1, col),
                    (row, col+1)]
        elif row == 0:
            return [(row+1, col),
                    (row, col+1),
                    (row, col-1)]
        elif col == 0:
            return [(row+1, col),
                    (row, col+1),
                    (row-1, col)]
        else:
            return [(row+1, col),
                    (row, col+1),
                    (row-1, col),
                    (row, col-1)]
        
        
    def step(self):
        a = np.copy(self.array)
        
        for row in xrange(self.n):
            for col in xrange(self.n):
                z = self.array[row, col]
                if z >= 4:
                    a[row, col] -= 4
                    for cell in self.neighbors(row, col):
                        try: a[cell] += 1
                        except IndexError: pass
        self.array = a
    
    def seed(self):
        seed = tuple(np.random.randint(self.n, size=2))
        self.array[seed] += 1
        
    def disturb(self, seed='random'):
        """
        seed is either random, or a tuple (row, col)"""
        previous =  self.array.copy()
        if seed=='random':
            seed = tuple(np.random.randint(self.n, size=2))
        self.array[seed] += 1

        completion_steps = self.stabilize()
        cluster_size = np.sum(previous != self.array)
        return cluster_size, completion_steps
        
    
    def stabilize(self):
        i = 0
        while True:
            previous = self.array
            self.step()
            i+=1
            if np.all(previous == self.array) and i>0:
                break
        return i
    
    def is_stable(self):
        return np.all(self.array<4)

class SandViewer(object):
    def __init__(self, sand, interval=200, cmap=matplotlib.cm.Greys):    
        self.cmap = cmap
        self.interval = interval
        self.sand = sand
        
    def frame_generator(self):
        counter=0
        while not self.sand.is_stable():
            yield self.i
            self.i += 1
            
    
    def seed_frame_generator(self, seeds=10):
        for s in xrange(seeds):
            iter = self.frame_generator()
            if self.i>0:
                self.sand.seed()
                print "seeded", self.i
            counter = 0
            while True:
                try:
                    nxt = iter.next()
                    print 'nxt', nxt, self.i
                    yield nxt
                except StopIteration:
                    print "got stop",nxt
                    break
            print 'seed',s,'i=',self.i
            
    def run(self, steps=50):
        from matplotlib import animation

        if steps=="complete":
            steps=self.frame_generator()
        if steps=='seed':
            steps=self.seed_frame_generator()
        self.steps = steps
        
        self.fig = plt.figure()
        self.ax = plt.axes()
        a = self.sand.array
        self.mesh = self.ax.pcolormesh(a, cmap=self.cmap)
        anim = animation.FuncAnimation(self.fig,
                                       self.animate,
                                       init_func=self.init,
                                       frames=self.steps, 
                                       interval=self.interval, 
                                       blit=True)
        return anim
    
    def init(self, verbose=True):
        self.i = 0
        if verbose: print "starting animation"
        a = self.sand.array
        self.mesh = self.ax.pcolormesh(a, cmap=self.cmap,
                                       vmin=0, vmax=self.sand.k + 1)
        self.fig.colorbar(self.mesh)
        return self.mesh, 
    
    def animate(self, i ):
        if i > 0:
            self.sand.step()
        a = self.sand.array
#         self.mesh = self.ax.pcolormesh(a, cmap=self.cmap)
        self.mesh.set_array(a.ravel())
        return self.mesh,

def fit_D2(inv_Es, Ns):
    """optimized to get rid of log0s"""
    inv_Es = inv_Es[:]
    Ns = Ns[:]
    assert len(inv_Es) == len(Ns)
    
    while True:
        try:
            if len(Ns) == 0:
                raise ValueError('could not fit')
            x = np.log10(inv_Es)
            y = np.log10(Ns)
        except:
            inv_Es.pop(0)
            Ns.pop(0)
            continue
        
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            break
        inv_Es.pop(0)
        Ns.pop(0)
          
    z = np.polyfit(x, y, 1)
    return z, inv_Es, np.power(inv_Es, z[0]) *np.power(10, z[1])

def calculate_fractal_dimension(a, title='', plot=True):
    """currently only works for square arrays, a"""
    from collections import deque
    height, width = a.shape

    #side lengths:
    side_lengths = deque()
    sl = 1
    while sl <= width:
        if width%sl==0:
            side_lengths.appendleft(sl)
        sl+=1
    inv_Es = [1.0/s for s in side_lengths]

    #Ns
    Ns = []
    for sl in side_lengths:
        N = 0
        for row in xrange(height):
            for col in xrange(width):
                if col % width == 0: continue #end
                if col % sl != 0: continue
                test = np.sum(a[row:row+sl, col:col+sl]) >= sl**2/2
                N += test
        Ns.append(N)
    
    z, fit_xs, fit_ys = fit_D2(inv_Es, Ns)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(inv_Es, Ns)
        ax.plot(fit_xs, fit_ys, label='D=%g'%z[0])
        ax.set_yscale('log')
        ax.set_ylabel('$N(\epsilon)$')
        ax.set_xscale('log')
        ax.set_xlabel('$1/\epsilon$')
        ax.set_title('Fractal Dimension Plot: %s'%title)
        ax.legend(loc='best')
    return z[0]

class SandFractalCalculator(object):
    
    def __init__(self, sand):
        self.sand =  sand
        self.get_minimally_stable()
    
    def get_minimally_stable(self):
        sand_array = self.sand.array
        a = np.zeros(sand.array.shape)
        
        for row in xrange(self.sand.n):
            for col in xrange(self.sand.n):
                neighbors = self.sand.neighbors(row, col)
                for n in neighbors:
                    try: 
                        if sand_array[n] >= 3: a[row,col] = 1
                    except IndexError: pass #neighbors returns cells not in array
        self.array = a
        
    def show(self):
        print self.array
        fig, ax = plt.subplots()
        ax.pcolormesh(self.array, cmap=matplotlib.cm.Greys)
        plt.show()
        
    def calculate(self):
        calculate_fractal_dimension(self.array)

                    



if __name__ == '__main__':

	sand = SandPile(n=50, k=10)
	sv = SandViewer(sand)
	# anim = sv.run('seed')
	# plt.show()
	
	sand.stabilize()
	for i in range(1000):
		if i%100==0: print i

		sand.disturb()
	sfc = SandFractalCalculator(sand) 
	sfc.calculate()
	sfc.show()

	plt.show()
