import torch
import torch.optim as optim

def optimize_latent_vector(score_model, device, latent_vector_init, n_iterations=100, learning_rate=0.01):
    latent_vector = latent_vector_init.clone().detach().to(device).requires_grad_(True)
    optimizer = optim.Adam([latent_vector], lr=learning_rate)

    for i in range(n_iterations):
        optimizer.zero_grad()
        score = score_model(latent_vector)
        loss = -torch.mean(score)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}, Score: {-loss.item()}")

    final_score = score_model(latent_vector)
    return latent_vector, final_score
