import ANNarchy as ann

baseline_dopa = ann.Constant('baseline_dopa', 0.1)
baseline_snr = ann.Constant('baseline_snr', 1.0)

ann.add_function('logistic(x) = 0.5 + 1 / (1 + exp(-3.5 * (x - 1)))')

# Neuron definitions
PoolingNeuron = ann.Neuron(
    parameters="""
        r_scale = 1.0 : population
    """,
    equations="""
        r = r_scale * sum(exc)
    """
)

OutputNeuron = ann.Neuron(
    equations="""
        r = if sum(norm) > 0.0: sum(exc) / sum(norm) else: sum(exc)
    """
)

BaselineNeuron = ann.Neuron(
    parameters="""
        tau_up = 10.0 : population
        tau_down = 20.0 : population
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        base = baseline + noise * Uniform(-1.0,1.0): min=0.0
        dr/dt = if (baseline>0.01): (base-r)/tau_up else: -r/tau_down : min=0.0
    """,
    name="Baseline Neuron",
    description="Time-dynamic neuron with baseline to be set. "
)

LinearNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0: population
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline
        r = pos(mp) 
    """
)

StriatumD1Neuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0 : population
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(mod) * (sum(exc) - sum(inh)) + noise*Uniform(-1.0,1.0) + baseline
        r = if (mp > 1.0): logistic(mp)
            else: pos(mp)
    """
)

SNrNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline_snr
        r = mp : min = 0.0, init = baseline_snr
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        firing = 0 : population, bool
        factor_inh = 10.0 : population
    """,
    equations="""
        s_inh = sum(inh)
        aux = firing * pos(1.0 - s_inh) + (1-firing)*baseline_dopa  
        tau*dmp/dt + mp =  aux
        r = mp : min = 0.0
    """
)

# Synapse definitions
ReversedSynapse = ann.Synapse(
    parameters="""
        reversal = 1.2 : projection
    """,
    psp="""
        w*pos(reversal-pre.r)
    """,
    name="Reversed Synapse",
    description="Higher pre-synaptic activity lowers the synaptic transmission and vice versa."
)

# DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
PostCovarianceNoThreshold = ann.Synapse(
    parameters="""
        tau = 200.0 : projection
        tau_alpha = 100.0 : projection
        regularization_threshold = 0.7 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre = 0.0 : projection
        threshold_post = 0.05 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - threshold_pre)
        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod =  if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum
                    else: condition_0*DA_type*K_dip*dopa_sum
        delta = dopa_mod * trace - alpha*pos(post.r - mean(post.r) - threshold_post) : min = 0.0
        tau*dw/dt = delta : min = 0.0
    """
)

# Inhibitory synapses STRD1 -> SNr
PreCovariance_inhibitory = ann.Synapse(
    parameters="""
        tau=100.0 : projection
        tau_alpha=20.0 : projection
        regularization_threshold = 0.4 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre = 0.05 : projection
        threshold_post = 0.05 : projection
        negterm = 1 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt + alpha = pos(-post.mp + regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r - threshold_post)
        aux = if (trace>0): negterm else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        delta = dopa_mod * trace - alpha * pos(mean(post.r) - post.r - threshold_post) : min = 0.0
        tau*dw/dt = delta : min = 0.0
    """
)

DAPrediction = ann.Synapse(
    parameters="""
        tau = 200.0 : projection
        threshold = 0.1 : projection
   """,
   equations="""
       aux = if (post.mp>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r) - threshold)
       tau*dw/dt = delta : min = 0.0
   """
)

CorticalLearning = ann.Synapse(
    parameters="""
        tau = 10000. : projection
        rho = 1. : projection
        threshold_pre = 0.0 : projection
    """,
    equations="""
        tau * dw/dt = pos(pre.r - threshold_pre) * post.r - rho * post.r
    """,
    description="STDP rule for inhibitory synapses introduced by Vogels et al. (2011)."
)

LearningMT = ann.Synapse(
    parameters ="""
        LearnTau = 1000. : projection
        minweight = 0.0 : projection
        alpha = 1.0 : projection
    """,
    equations = """            
        LearnTau * dw/dt = (pre.r - mean(pre.r)) * post.r - alpha * post.r^2 * w : min = minweight, init = 0.0
    """
)

NewAntihebb = ann.Synapse(
    parameters ="""
        TauAH = 10000. : projection
        alpha = 1 : projection
        gamma = 1 : projection
        rho = 0.06 : projection        
    """,
    equations = """    
        TauAH * dw/dt = pre.r * post.r - pre.r * rho * (gamma + alpha * w) : min = 0.0, init =0.0
    """
)

MiehlExc = ann.Synapse(
    parameters="""
		tau_W = 10000 : projection
        alpha = 0.5  : projection
	""",
    equations="""
    	tau_W * dw/dt = pre.r * post.r * (post.r - alpha) : min = 0.0
	""",
)

OjaLearningRule = ann.Synapse(
    parameters="""
    eta = 0.01 : projection
    alpha = 1.0 : projection
    """,
    equations="""
     dw/dt = eta * ( pre.r * post.r - alpha * post.r^2 * w ) : min=0.0
    """
)

BCMLearningRule = ann.Synapse(
    parameters = """
        eta = 0.01 : projection
        tau = 2000.0 : projection
    """,
    equations = """
        tau * dtheta/dt + theta = post.r^2 : postsynaptic, exponential
        dw/dt = eta * post.r * (post.r - theta) * pre.r : min=0.0, explicit
    """
)
