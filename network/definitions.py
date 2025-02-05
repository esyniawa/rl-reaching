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
        tau = 20.0 : population
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
        tau = 20.0 : population
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
        tau = 20.0 : population
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline_snr
        r = pos(mp) : init = baseline_snr
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 20.0 : population
        firing = 0 : population, bool
        factor_inh = 10.0 : population
        error_threshold = 0.0 : population
    """,
    equations="""
        deviation = sum(target) - sum(vl_rates)  # Compare CM output with SNr output
        factor_da = if deviation >= error_threshold: 1.0 else: 0.0
        mp = if firing:
                factor_da * pos(1.0 - sum(inh_rpe)) + (1.0 - factor_da)*(baseline_dopa - factor_inh*deviation)  
            else: 
                baseline_dopa
        tau*dr/dt + r =  pos(mp)
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
        tau = 150.0 : projection
        tau_alpha = 1000.0 : projection
        regularization_threshold = 0.9 : projection
        K_burst = 1.0 : projection
        K_dip = 0.8 : projection
        DA_type = 1 : projection
        threshold_pre = 0.05 : projection
        threshold_post = 0.0 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - threshold_pre)
        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod =  if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum
                    else: condition_0*DA_type*K_dip*dopa_sum
        delta = dopa_mod * trace - alpha*pos(post.r - mean(post.r) - threshold_post)
        tau*dw/dt = delta : min = 0.0
    """
)

DAPrediction = ann.Synapse(
    parameters="""
        tau = 1000.0 : projection
        threshold = 0.05 : projection
   """,
   equations="""
       aux = if (post.mp>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r) - threshold)
       tau*dw/dt = delta : min = 0.0
   """
)
