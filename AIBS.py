
# coding: utf-8

# In[1]:

try:
    import rickpy
    rickpy.use_dev_packages(['rickpy','scidash/sciunit','scidash/neuronunit','neuroml/pyNeuroML'])
    import imp
    imp.reload(rickpy)
    print("Using local development versions of packages")
except ImportError:
    print("Using packages from system path")


# In[2]:

#get_ipython().magic('matplotlib notebook')
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import sciunit
import neuronunit
from neuronunit import aibs
#from neuronunit.models.reduced import ReducedModel

import pdb

from neuroml import nml
from pyneuroml import pynml



def init_nrn_model():#do this once only.
    '''
    Take a declarative model description, and convert it into an implementation, stored in a pyhoc file.
    import the pyhoc file thus dragging the neuron variables into memory/python name space.
    Since this only happens once outside of the optimization loop its a tolerable performance hit.
    '''
    #could this be a shorter root to NeuroML2/LEMS_2007One.xml saving the user from needing to supply it given NeuroML2 may already be installed?
    #izc=nml.nml.Izhikevich2007Cell
    IZHIKEVICH_PATH = os.getcwd()
    LEMS_MODEL_PATH = os.path.join(IZHIKEVICH_PATH,'NeuroML2/LEMS_2007One.xml')                        
    # github.com/OpenSourceBrain/IzhikevichModel.
    DEFAULTS={}
    DEFAULTS['v']=True
    pynml.run_lems_with_jneuroml_neuron(LEMS_MODEL_PATH, 
                                      skip_run=False,
                                      nogui=False, 
                                      load_saved_data=False, 
                                      plot=False, 
                                      show_plot_already=False, 
                                      exec_in_dir = ".",
                                      only_generate_scripts = True,
                                      verbose=DEFAULTS['v'],
                                      exit_on_fail = True)
    init_nrn_model()

def update_nrn_param(param_dict):
    #TODO find out the python3 syntax for dereferencing key value pairs.
    #Below syntax is stupid, but how to just get key generically without for knowledge of its name and without iterating?
    key=[ key for key in param_dict.keys() ][0]
    value=[ value for value in param_dict.values() ][0]
    print(key)
    print(value)
    pdb.set_trace()
    evalstring='neuron.hoc.execute("m_RS_RS_pop[0].'+str(key)+'='+str(value)+'")'
    eval(evalstring)
    neuron.hoc.execute('forall{ psection() }')
   
from NeuroML2 import LEMS_2007One_nrn     
from NeuroML2.LEMS_2007One_nrn import NeuronSimulation
neuron.load_mechanisms(os.getcwd()+'/NeuroML2')
ns = NeuronSimulation(tstop=1600, dt=0.0025)
neuron.hoc.execute('forall{ psection() }')
neuron.psection(neuron.nrn.Section())
param_dict={}
for vr in np.linspace(-75,-50,6):
    param_dict['vr']=vr               
    update_nrn_param(param_dict)




#neuron.hoc.execute('RS_pop[0].L = 11.1')
                      
#pdb.set_trace()

from neuronunit import models
from neuronunit.models.reduced import ReducedModel
vr=-65.0
model = ReducedModel(LEMS_MODEL_PATH, name='V_rest=%dmV' % vr, attrs={'//izhikevich2007Cell': {'vr':'%d mV' % vr} })
model.run()
import pdb
pdb.set_trace()
#from neuronunit
# import SimpleModel
#sm = SimpleModel()
# In[6]:

import quantities as pq
from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []

dataset_id = 354190013  # Internal ID that AIBS uses for a particular Scnn1a-Tg2-Cre 
                        # Primary visual area, layer 5 neuron.
observation = aibs.get_observation(dataset_id,'rheobase')
tests += [nu_tests.RheobaseTest(observation=observation)]
    
test_class_params = [(nu_tests.InputResistanceTest,None),
                     (nu_tests.TimeConstantTest,None),
                     (nu_tests.CapacitanceTest,None),
                     (nu_tests.RestingPotentialTest,None),
                     (nu_tests.InjectedCurrentAPWidthTest,None),
                     (nu_tests.InjectedCurrentAPAmplitudeTest,None),
                     (nu_tests.InjectedCurrentAPThresholdTest,None)]

for cls,params in test_class_params:
    observation = cls.neuroelectro_summary_observation(neuron)
    tests += [cls(observation,params=params)]
    
def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']
    for i in [5,6,7]:
        print(tests[i])
        # Set current injection to just suprathreshold
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 
    
hooks = {tests[0]:{'f':update_amplitude}}
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)


# In[7]:

model = ReducedModel(LEMS_MODEL_PATH,name='vanilla')
suite.judge(model)


# In[8]:

models = []

for vr in np.linspace(-75,-50,6):


    model = ReducedModel(LEMS_MODEL_PATH, 
                         name='V_rest=%dmV' % vr, 
                         attrs={'//izhikevich2007Cell':
                                    {'vr':'%d mV' % vr}
                               })
    #model.skip_run = True
    models.append(model)
suite.set_verbose(False) # Turn off most print statements.  
score_matrix = suite.judge(models)
score_matrix.show_mean = True
score_matrix.sortable = True
score_matrix


# In[9]:

score_matrix.view()


# In[10]:

import matplotlib as mpl
mpl.rcParams['font.size'] = 20
vm = score_matrix[tests[3]][4].related_data['vm'].rescale('mV') # Plot the rheobase current (test 3) 
                                                                # from v_rest = -55 mV (model 4)
ax = plt.gca()
ax.plot(vm.times,vm)
y_min = float(vm.min()-5.0*pq.mV)
y_max = float(vm.max()+5.0*pq.mV)
ax.set_xlim(0,1.6)
ax.set_ylim(y_min,y_max)
ax.set_xlabel('Time (s)',size=24)
ax.set_ylabel('Vm (mV)',size=24);
plt.tight_layout()


# In[8]:

"""
for a in np.linspace(0.015,0.045,2):
    for b in np.linspace(-3.5,-0.5,2):
        for C in np.linspace(50,150,3):
            for k in np.linspace(0.4,1.0,3):
                model = ReducedModel(LEMS_MODEL_PATH, 
                             name='a=%.3fperms_b=%.1fnS_C=%dpF_k=%.2f' % (a,b,C,k), 
                             attrs={'//izhikevich2007Cell':
                                        {'b':'%.1f nS' % b,
                                         'a':'%.3f per_ms' % a,
                                         'C':'%d pF' % C,
                                         'k':'%.2f nS_per_mV' % k,
                                         'vr':'-68 mV',
                                         'vpeak':'45 mV'}
                                   })
                #model.skip_run = True
                models3.append(model)
score_matrix3 = suite.judge(models3, verbose=False)
score_matrix3.show_mean = True
score_matrix3.sortable = True
score_matrix3
""";


# In[93]:

models2 = []
for i,a in enumerate(np.linspace(0.015,0.045,7)):
    for j,b in enumerate(np.linspace(-3.5,-0.5,7)):
        model = ReducedModel(LEMS_MODEL_PATH, 
                     name='a=%.3fperms_b=%.1fnS' % (a,b), 
                     attrs={'//izhikevich2007Cell':
                                {'b':'%.1f nS' % b,
                                 'a':'%.3f per_ms' % a,
                                 'C':'150 pF',
                                 'k':'0.70 nS_per_mV',
                                 'vr':'-68 mV',
                                 'vpeak':'45 mV'}
                           })
        #model.skip_run = True
        models2.append(model)
score_matrix2 = suite.judge(models2)
score_matrix2.show_mean = True
score_matrix2.sortable = True
score_matrix2.view()


# In[94]:

import matplotlib as mpl
mpl.rcParams['font.size'] = 18
heatmap = np.zeros((7,7))
for i,a in enumerate(np.linspace(0.015,0.045,7)):
    for j,b in enumerate(np.linspace(-3.5,-0.5,7)):
        for model in score_matrix2.models:
            if model.name == 'a=%.3fperms_b=%.1fnS' % (a,b):
                heatmap[i,j] = 20*(score_matrix2[model].mean() - 0.8070)+0.8070#[tests[0]].score
#heatmap[2,0] = np.nan
plt.pcolor(heatmap,cmap='magma')
plt.yticks(np.arange(7)+0.5,np.linspace(0.015,0.045,7))
plt.ylabel('Izhikevich Parameter $a$')
plt.xticks(np.arange(7)+0.5,np.linspace(-3.5,-0.5,7))
plt.xlabel('Izhikevich Parameter $b$')
cbar = plt.colorbar()
cbar.set_label('Mean Test Score',size=15)
cbar.ax.tick_params(labelsize=15) 
plt.tight_layout()
np.save('heatmap',heatmap)


# In[11]:

"""
from neuronunit.tests.dynamics import TFRTypeTest,BurstinessTest

is_bursty = BurstinessTest(observation={'cv_mean':1.5, 'cv_std':1.0})
score_matrix2 = is_bursty.judge(models)
score_matrix2
""";


# In[12]:

"""
#rickpy.refresh_objects(locals(),modules=None)
rickpy.refresh_objects(locals().copy(),modules=['sciunit','neuronunit'])
isinstance(tests[0],sciunit.Test) # Should print True if successful
""";

