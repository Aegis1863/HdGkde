import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def hd_gaussian_kde(data, data_scope, bandwidth=0.1, grid_size=100, device='cpu'):
    ''' support high dimension kde
    ---
    `data`: tensor, shape=(batch, m, n) \\
    `data_scope`: tensor, shape=(n, 2)  \\
    `bandwidth`: int | tensor, shape=(n,) \\
    '''
    b, m, n = data.shape  # `b` is the batch size, `m` is the number of small samples per sample, and `n` is the dimension of the small samples.
    entropy_values = torch.zeros(b, device=device)
    kde_values = []
    
    grid_n = torch.meshgrid(*[torch.linspace(*i_scope, grid_size, device=device) for i_scope in data_scope], indexing='ij')
    grid = torch.stack(grid_n, dim=-1).reshape(-1, n).to(device)
    
    for i in range(b):
        sample_points = data[i]
        kde = torch.zeros(grid.shape[0], device=device)
        for j in range(m):
            mean = sample_points[j]
            cov_matrix = (torch.eye(n, device=device) * bandwidth)
            mvn = MultivariateNormal(mean, cov_matrix)
            kde += mvn.log_prob(grid).exp()
        kde /= m
        kde = kde.view(*[grid_size for _ in range(n)])
        
        delta_v = (2 / grid_size) ** n  # Grid cell area / volume, etc.
        kde_sum = torch.sum(kde) * delta_v
        kde_norm = kde / kde_sum
        
        kde_values.append(kde_norm)
        
        entropy = -torch.sum(kde_norm * torch.log(kde_norm + 1e-12)) * delta_v
        entropy_values[i] = entropy

    # Returns the density value of the grid point together with the density value of the coordinate point
    # Can be delete if don't need specific kde values
    grid_values = []
    for i in range(b):
        kde_value = kde_values[i].reshape(-1).detach().cpu().numpy()
        grid_value = grid.cpu().numpy()
        grid_values.append((grid_value, kde_value))
    
    return grid_values, entropy_values, grid_n
