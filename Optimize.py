
# coding: utf-8

# In[17]:

#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')
#get_ipython().magic('matplotlib notebook')
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import sciunit
import neuronunit
from neuronunit import aibs
from neuronunit.models.reduced import ReducedModel
import pdb

# In[16]:

# This example is from https://github.com/OpenSourceBrain/IzhikevichModel.
model_path = os.getcwd()+str('/neuronunit/software_tests/NeuroML2') # Replace this the path to your 
                                                                       # working copy of 
                                                                       # github.com/OpenSourceBrain/IzhikevichModel.  
#LEMS_MODEL_PATH = os.path.join(IZHIKEVICH_PATH,)
file_path=model_path+str('/LEMS_2007One.xml')


from neuronunit.models import backends
#model = ReducedModel(IZHIKEVICH_PATH+str('/LEMS_2007One.xml'),name='vanilla')
#dir(backends.NeuronBackend)
model=backends.NEURONBackend(file_path,model_path,name='vanilla')
model=model.load_model()


#some testing of functionality
#TODO rm later.

#NeuronObject=backends.NEURONBackend(IZHIKEVICH_PATH,name='vanilla')


# In[8]:

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
    #pdb.set_trace()   

    
def update_amplitude(test,tests,score):
    rheobase = score.prediction['value']
    pdb.set_trace()
    for i in [5,6,7]:
        print(tests[i])
        # Set current injection to just suprathreshold
        print(type(rheobase))
        tests[i].params['injected_square_current']['amplitude'] = rheobase*1.01 

#update_amplitude(test,tests,score)

hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
#update amplitude at the location in sciunit thats its passed to, without any loss of generality.
suite = sciunit.TestSuite("vm_suite",tests,hooks=hooks)


# In[4]:
#model = ReducedModel(IZHIKEVICH_PATH+str('/LEMS_2007One.xml'),name='vanilla')
#model = ReducedModel(LEMS_MODEL_PATH,name='vanilla',backend='NEURONbackend')


# In[5]:

SUO = '/home/mnt/scidash/sciunitopt'
if SUO not in sys.path:
    sys.path.append(SUO)


# In[6]:

from types import MethodType
def optimize(self,model,rov):
    best_params = None
    best_score = None
    from sciunitopt.deap_config_simple_sum import DeapCapsule
    dc=DeapCapsule()
    pop_size=12
    ngen=5                                  
    pop = dc.sciunit_optimize(suite,file_path,pop_size,ngen,rov,
                                                         NDIM=2,OBJ_SIZE=2,seed_in=1)
                                                         
            #sciunit_optimize(self,test_or_suite,model_path,pop_size,ngen,rov,
            #                  NDIM,OBJ_SIZE,seed_in):

                                                         
            #sciunit_optimize(self,suite,LEMS_MODEL_PATH,pop_size,ngen,rov,
    #                                                   NDIM=1,OBJ_SIZE=1,seed_in=1):

    return pop#(best_params, best_score, model)

my_test = tests[0]
my_test.verbose = True
my_test.optimize = MethodType(optimize, my_test) # Bind to the score.


# In[7]:

rov = np.linspace(-67,-40,1000)
pop = my_test.optimize(model,rov)
#print('pareto front top value in pf hall of fame')
#print('best params',best_params,'best_score',best_score, 'model',model)


# In[13]:

print("%.2f mV" % np.mean([p[0] for p in pop]))


NeuronObject=backends.NEURONBackend(LEMS_MODEL_PATH)
NeuronObject.load_model()#Only needs to occur once
#NeuronObject.update_nrn_param(param_dict)
#NeuronObject.update_inject_current(stim_dict)
'''
brute force optimization:
for comparison
#print(dir(NeuronObject))
for vr in iter(np.linspace(-75,-50,6)):
    for i,a in iter(enumerate(np.linspace(0.015,0.045,7))):
        for j,b in iter(enumerate(np.linspace(-3.5,-0.5,7))):
            for k in iter(np.linspace(100,200,4)):
                param_dict={}#Very important to redeclare dictionary or badness.
                param_dict['vr']=vr

                param_dict['a']=str(a) 
                param_dict['b']=str(b)               
                param_dict['C']=str(150)
                param_dict['k']=str(0.70) 
                param_dict['vpeak']=str(45)                      
                             
                NeuronObject.update_nrn_param(param_dict)
                stim_dict={}
                stim_dict['delay']=200
                stim_dict['duration']=500
                stim_dict['amplitude']=k#*100+150

                NeuronObject.update_inject_current(stim_dict)
                NeuronObject.local_run()
                vm,im,time=NeuronObject.out_to_neo()
                print('\n')
                print('\n')
                print(vm.trace)
                print(time.trace)
                print(im.trace)
                print('\n')
                print('\n')
'''
