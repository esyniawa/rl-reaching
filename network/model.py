from .params import parameters

from .connections import *
from .definitions import *

ann.setup(num_threads=8)

# input populations
PM = ann.Population(geometry=parameters['dim_pm'], neuron=BaselineNeuron, name='PM')
S1 = ann.Population(geometry=parameters['dim_s1'], neuron=BaselineNeuron, name='S1')

SNc = ann.Population(geometry=2, neuron=DopamineNeuron, name='SNc')

# transmission populations into putamen
CM = ann.Population(geometry=parameters['dim_motor'], neuron=BaselineNeuron, name='CM')
CM.tau_up = 20.
CM.noise = 0.01

# CBGT Loop (putamen)
StrD1 = ann.Population(geometry=parameters['dim_str'], neuron=StriatumD1Neuron, name='StrD1')
StrD1.noise = 0.0

SNr = ann.Population(geometry=parameters['dim_bg'], neuron=SNrNeuron, name='SNr', stop_condition='r<0.1')
SNr.noise = 0.01

VL = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='VL')
VL.noise = 0.01
VL.baseline = ann.get_constant('baseline_snr') - 0.065

M1 = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='M1')
M1.tau = 50.
M1.noise = 0.0
M1.baseline = 0.0

# output population
Output_Pop_Shoulder = ann.Population(geometry=3, neuron=OutputNeuron, name='Output_Shoulder')
Output_Pop_Elbow = ann.Population(geometry=3, neuron=OutputNeuron, name='Output_Elbow')

# Projections
CM_StrD1 = {}
w_CM_str = w_one_to_ones(preDim=parameters['dim_motor'][0], postDim=tuple(list(parameters['dim_s1']) + [parameters['dim_motor'][0]]), weight=parameters['strength_cm'])
for i, subset_key in enumerate(parameters['subsets_str']):
    interval = parameters['subsets_str'][subset_key]
    CM_StrD1[subset_key] = ann.Projection(pre=CM[:, i], post=StrD1[:, :, interval[0]:interval[1]], target='exc', name=f"CM_StrD1_{subset_key}")
    CM_StrD1[subset_key].connect_from_matrix(w_CM_str)

S1_StrD1 = ann.Projection(pre=S1, post=StrD1, target='mod')
w_s1 = w_2D_to_3D_S1(preDim=parameters['dim_s1'], postDim=StrD1.geometry)
S1_StrD1.connect_from_matrix(w_s1)

PM_StrD1 = ann.Projection(pre=PM, post=StrD1, target='exc', synapse=PostCovarianceNoThreshold, name='PM_D1')
PM_StrD1.connect_all_to_all(0.0)

StrD1_SNr = {}
for i, subset_key in enumerate(parameters['subsets_str']):
    interval = parameters['subsets_str'][subset_key]
    w_Str_SNr = w_pooling(preDim=StrD1[:, :, interval[0]:interval[1]].geometry, poolingDim=-1, weight=parameters['strength_snr'])
    StrD1_SNr[subset_key] = ann.Projection(pre=StrD1[:, :, interval[0]:interval[1]], post=SNr[:, i],
                                           target='inh', name=f'D1_SNr_{subset_key}')
    StrD1_SNr[subset_key].connect_from_matrix(w_Str_SNr)

# dopa connections
SNc_SNr = ann.Projection(pre=SNc, post=SNr, target='dopa')
SNc_SNr.connect_all_to_all(1.0)

SNc_StrD1 = ann.Projection(pre=SNc, post=StrD1, target='dopa')
SNc_StrD1.connect_all_to_all(1.0)

SNr_VL = ann.Projection(pre=SNr, post=VL, target='inh')
SNr_VL.connect_one_to_one(1.0)

# #TODO: rewrite
# for layer in range(2):
#     VL_M1 = ann.Projection(pre=VL[:, layer], post=M1[:, layer], target='exc')
#     w_vl_m1 = connect_gaussian_circle(Dim=parameters['dim_bg'][0], scale=parameters['sig_m1'],
#                                       sd=parameters['sig_vl_m1'], A=parameters['A_vl_m1'])
#     VL_M1.connect_from_matrix(w_vl_m1)

VL_M1 = ann.Projection(pre=VL, post=M1, target='exc')
VL_M1.connect_one_to_one(weights=0.8)

# Output projection
PopCode_shoulder = ann.Projection(pre=M1[:, 0], post=Output_Pop_Shoulder, target='exc')
w_out = pop_code_output(preferred_angles=parameters['motor_orientations'])
PopCode_shoulder.connect_from_matrix(w_out)

PopCode_elbow = ann.Projection(pre=M1[:, 1], post=Output_Pop_Elbow, target='exc')
w_out = pop_code_output(preferred_angles=parameters['motor_orientations'])
PopCode_elbow.connect_from_matrix(w_out)

# normalize PopCode
PopCode_norm_shoulder = ann.Projection(pre=M1[:, 0], post=Output_Pop_Shoulder[0], target='norm')
PopCode_norm_shoulder.connect_all_to_all(1.0)

PopCode_norm_elbow = ann.Projection(pre=M1[:, 1], post=Output_Pop_Elbow[0], target='norm')
PopCode_norm_elbow.connect_all_to_all(1.0)

# Reward prediction
StrD1_SNc = {}
for i, subset_key in enumerate(parameters['subsets_str']):
    interval = parameters['subsets_str'][subset_key]
    StrD1_SNc[subset_key] = ann.Projection(pre=StrD1[:, :, interval[0]:interval[1]], post=SNc[i],
                                           target='inh', name=f'D1_SNc_{subset_key}', synapse=DAPrediction)
    StrD1_SNc[subset_key].connect_all_to_all(0.0)

# Laterals
# SNr_SNr = ann.Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse)
# wSNr_SNr = laterals_layerwise(Dim=SNr.geometry, axis=0, weight=0.1)
# SNr_SNr.connect_from_matrix(wSNr_SNr)

# M1_M1 = ann.Projection(pre=M1, post=M1, target='inh')
# wM1_M1 = laterals_layerwise(Dim=M1.geometry, axis=0, weight=0.0)
# M1_M1.connect_from_matrix(wM1_M1)

# laterals on the last dimension
# StrD1_StrD1 = ann.Projection(pre=StrD1, post=StrD1, target='inh')
# wD1_D1 = laterals_layerwise(Dim=StrD1.geometry, axis=2, weight=0.2, subset_dict=parameters['subsets_str'])
# StrD1_StrD1.connect_from_matrix(wD1_D1)
