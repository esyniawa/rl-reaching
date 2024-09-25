import os
import numpy as np
import matplotlib.pyplot as plt


sims = (1, 2)
error = {}

for sim in sims:
    for root, dirs, _ in os.walk(f'results/test_ppo_{sim}'):
        for d in dirs:
            if sim == sims[0]:
                error[d] = []

            data = np.load(os.path.join(root, d) + '/error.npy')
            error[d].append(data)

results = {}
for i, key in enumerate(error):
    result = np.array(error[key])
    results['mean_shoulder_' + key] = np.abs(result[:, :, 0]).mean()
    results['std_shoulder_' + key] = np.abs(result[:, :, 0]).std()

    results['mean_elbow_' + key] = np.abs(result[:, :, 1]).mean()
    results['std_elbow_' + key] = np.abs(result[:, :, 1]).std()

n_training_trails = (1000, 2000)
error = np.zeros((2, 2, len(n_training_trails)))

for i, trials in enumerate(n_training_trails):
    error[0, 0, i] = results['mean_shoulder_model_' + str(trials)]
    error[0, 1, i] = results['std_shoulder_model_' + str(trials)]

    error[1, 0, i] = results['mean_elbow_model_' + str(trials)]
    error[1, 1, i] = results['std_elbow_model_' + str(trials)]

fig, axs = plt.subplots(ncols=2, figsize=(6, 3), sharey=True)
axs[0].plot(error[0, 0, :], label='$\\theta_{shoulder}$', color='orange')
axs[1].plot(error[1, 0, :], label='$\\theta_{elbow}$', color='blue')

axs[0].fill_between(np.arange(0, len(n_training_trails)), error[0, 0, :] - error[0, 1, :], error[0, 0, :] + error[0, 1, :], color='orange',  alpha=0.2)
axs[1].fill_between(np.arange(0, len(n_training_trails)), error[1, 0, :] - error[1, 1, :], error[1, 0, :] + error[1, 1, :], color='blue', alpha=0.2)

axs[0].set_xlabel('# of training trials in thousands'), axs[1].set_xlabel('# of training trials in thousands')
axs[0].set_ylabel('Error in [°]')

axs[0].legend(), axs[1].legend()
axs[0].set_ylim(0, np.pi), axs[1].set_ylim(0, np.pi)
axs[0].set_xticks(np.arange(0, len(n_training_trails)), np.array(n_training_trails)/1000)
axs[1].set_xticks(np.arange(0, len(n_training_trails)), np.array(n_training_trails)/1000)

plt.savefig('ppo_error.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()