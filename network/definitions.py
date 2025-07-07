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
        feedback = sum(target)
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
        r = mp : min = 0.0, init = baseline_snr
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 20.0 : population
        firing = 0 : population, bool
        factor_inh = 10.0 : population
        rate = 0.0
    """,
    equations="""
        s_inh = sum(inh)
        aux = firing * pos(rate - s_inh) + (1-firing)*baseline_dopa  
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

PostDeltaRule = ann.Synapse(
    parameters="""
        tau = 100.0 : projection
        learning_rate = 2.0 : projection
        decay_rate = 1.0 : projection
    """,
    equations="""
        # dopamine modulation of learning rate
        dopa_signal = post.sum(dopa) - baseline_dopa: min = 0.0
        
        # Delta rule: delta w = lr * error * input
        error = post.feedback - post.r
        delta_ltp = if (dopa_signal > 0.0): learning_rate * dopa_signal * error * pre.r else: 0.0
        delta_ltd = if (dopa_signal < 0.0): decay_rate * dopa_signal * w else: 0.0
        
        tau*dw/dt = delta_ltp + delta_ltd : min = 0.0
    """
)

DAPrediction = ann.Synapse(
    parameters="""
        tau = 250.0 : projection
        threshold = 0.1 : projection
   """,
   equations="""
       aux = if (post.mp>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r) - threshold)
       tau*dw/dt = delta : min = 0.0
   """
)
