'''Usage:
import simple
h.run()
simple.show()

Sets up 5 models using default parameters in the .mod files
2 versions of 2003/2004 parameterization: freestanding (3a); in section (3b)
3 versions of 2007/2008 parameterization: freestanding (7a); in section (7b); in sec using wrapper class (7bw)
can graph u, v for any model
simple.show('v3a','v3b') # compare voltage output for the 2 versions of the 2003/2004 parameterization; will NOT be identical
simple.show('v7a','v7b','v7bw') # compare voltage output for 3 versions of the 2007 parameterization
'''

from neuron import h, gui
import numpy as np
import izhi2007Wrapper as izh07
import matplotlib
matplotlib.use('agg')
import pylab as plt

import pprint as pp
plt.ion()
fih = []

from neuronunit.models import ExternalModel

# make a 2007b (section) cell
#sec07b = h.Section()
#sec07b.L, sec07b.diam = 5, 6.3
#iz07b = h.Izhi2007b(0.5,sec=sec07b)
#iz07b.Iin = 70
#fih.append(h.FInitializeHandler(iz07b_init))

# make a 2007b (section) cell using the Wrapper
iz07bw = izh07.IzhiCell() # defaults to RS
iz07bw.izh.Iin = 70
def iz07b_init(): sec07b.v=-60
fih.append(h.FInitializeHandler(iz07bw.init))

# vectors and plot
def vtvec(vv): return np.linspace(0, len(vv)*h.dt, len(vv), endpoint=True)

def inject_square_current(current, section = None):
    """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
    Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
    where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
    Description: A parameterized means of applying current injection into defined
    Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

    Implementation:
    1. purge the HOC space, by calling reset_neuron()
    2. Redefine the neuronal model in the HOC namespace, which was recently cleared out.
    3. Strip away quantities representation of physical units.
    4. Translate the dictionary of current injection parameters into executable HOC code.

    """


    c = copy.copy(current)
    if 'injected_square_current' in c.keys():
        c = current['injected_square_current']
    c['delay'] = re.sub('\ ms$', '', str(c['delay'])) # take delay
    c['duration'] = re.sub('\ ms$', '', str(c['duration']))
    c['amplitude'] = re.sub('\ pA$', '', str(c['amplitude']))
    # NEURONs default unit multiplier for current injection values is nano amps.
    # to make sure that pico amps are not erroneously interpreted as a larger nano amp.
    # current injection value, the value is divided by 1000.
    amps=float(c['amplitude'])/1000.0 #This is the right scale.


    prefix = str('iz07bw.sec(0.5)')
    #'explicitInput_%s%s_pop0.' % (self.current_src_name,self.cell_name)
    define_current = []
    define_current.append(prefix+'amplitude=%s'%amps)
    define_current.append(prefix+'duration=%s'%c['duration'])
    define_current.append(prefix+'delay=%s'%c['delay'])
    for string in define_current:
        # execute hoc code strings in the python interface to neuron.
        h(string)

# run and plot
fig = None
def show (*vars):
  pp.pprint(recd.keys())
  global fig,tvec
  if fig is None: fig = plt.figure(figsize=(10,6), tight_layout=True)
  if len(vars)==0: vars=recd.keys()
  tvec=vtvec(recd['v7bw'][1])
  plt.clf()
  [plt.plot(tvec,v[1]) for x,v in recd.items() if x in vars]
  pp.pprint([list(v[1])[-5:] for x,v in recd.items() if x in vars])
  plt.xlim(0,h.tstop)

#h.run()
from neo.core import AnalogSignal
import quantities as qt
ms = qt.ms
mV = qt.mV

import pickle,os
from neuronunit.optimization import get_neab
electro_path = str('/home/jovyan/neuronunit/neuronunit/unit_test')+str('/pipe_tests.p')
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
tests, observation = electro_tests[0]


scores = {}
#import dask.bag as db

kl = list(izh07.type2007.keys())
E = ExternalModel()
E.inject_square_current = inject_square_current
for i in kl:
    iz07bw.reparam(type=i)
    print(iz07bw.izh.C,iz07bw.izh.k,iz07bw.izh.vr,iz07bw.izh.vt,iz07bw.izh.vpeak,iz07bw.izh.a,iz07bw.izh.b,iz07bw.izh.c,iz07bw.izh.d,iz07bw.izh.celltype)
    h.tstop=1250
    print(h.psection())
    v = []
    recd = {'u7bw':[iz07bw.izh._ref_u], 'v7bw':[iz07bw.sec(0.5)._ref_v]}
    for x,v in recd.items():
        v.append(h.Vector(h.tstop/h.dt+100))
        v[1].record(v[0])
    h.run()
    dt = h.dt
    print(np.shape(v[1]))
    vm = AnalogSignal(v[1],
                        units = mV,
                        sampling_period = dt * ms)
    E.set_membrane_potential(vm)

    for t in tests:
        score = t.judge(E)
        print(dir(E.inject_square_current))
        print(score)
        scores[str(t)] = score
        #observations[str(t)] = score.observation
        #predictions[str(t)] = score.prediction

    #print(E.get_membrane_potential())

    #show()
