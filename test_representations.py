import torch
import matplotlib.pyplot as plt
# from geomloss import SamplesLoss
import ot
import ot.plot
import numpy as np 


model1 = "stabilityai/stable-diffusion-xl-base-1.0"
model2 = "cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune"


mean_diffs = []



for itime in range(0, 50):


    representations1 = torch.load(f"hidden_reps/{model1}/input_t{itime}.pt").flatten(1).transpose(0, 1)
    representations2 = torch.load(f"hidden_reps/{model2}/input_t{itime}.pt").flatten(1).transpose(0, 1)

    n_samples = representations1.shape[0]
    a = torch.ones(n_samples) / n_samples  # Source weights
    b = torch.ones(n_samples) / n_samples


    reps1 = representations1.to(torch.float32)
    reps2 = representations2.to(torch.float32)

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    
    # representations1 = representations1[:10]
    # representations2 = representations2[:10]

    M = ot.dist(representations1, representations2, metric='sqeuclidean').to(torch.float32)


    # print(M)
    print(M.shape)

    reg = 0.01

    # Compute optimal transport using Sinkhorn algorithm
    print("Computing Sinkhorn transport plan...")
    pi = ot.sinkhorn(a, b, M, reg)

    print(pi.shape)

    

    quit()


    diff = torch.mean(torch.abs(representations1 - representations2) / torch.mean(abs(representations1)))

    mean_diffs.append(diff)


    T = torch.argmin(sinkhorn(representations1, representations2)).float()

    quit()




mean_diffs = torch.stack(mean_diffs, dim = 0)

# Plot the mean differences over time
plt.figure(figsize=(10, 6))
plt.plot(range(50), mean_diffs.numpy())
plt.title('Mean Differences Over Time')
plt.xlabel('Time')
plt.ylabel('Mean Absolute Relative Difference')
plt.savefig('outputs/mean_differences_over_time.png')









