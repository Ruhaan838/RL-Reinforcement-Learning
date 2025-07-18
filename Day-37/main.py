import torch
from policies import Policy
from meta_ppo import collect_trajectory, compute_loss
from tasks import sample_task
from copy import deepcopy
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta_policy = Policy(input_dim=7*7*3, action_dim=7).to(device)
optimizer = torch.optim.Adam(meta_policy.parameters(), lr=1e-3)

for iteration in range(1000):
    meta_loss = 0
    for _ in range(4):  # 4 tasks per meta-update
        env = sample_task()
        policy_clone = deepcopy(meta_policy)

        traj = collect_trajectory(env, policy_clone, steps=100, device=device)
        inner_loss = compute_loss(traj, policy_clone)
        policy_clone.load_state_dict(
            {name: param - 0.1 * grad
             for (name, param), grad in zip(policy_clone.named_parameters(),
                                            torch.autograd.grad(inner_loss, policy_clone.parameters(), create_graph=True))}
        )

        post_traj = collect_trajectory(env, policy_clone, steps=100, device=device)
        meta_loss += compute_loss(post_traj, meta_policy)

    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    print(f"[{iteration}] Meta Loss: {meta_loss.item():.3f}")
    
env = gym.make("MiniGrid-DoorKey-6x6-v0")
adapted_policy = deepcopy(meta_policy)

for _ in range(3):
    traj = collect_trajectory(env, adapted_policy, steps=100, device=device)
    loss = compute_loss(traj, adapted_policy)
    grads = torch.autograd.grad(loss, adapted_policy.parameters())
    for p, g in zip(adapted_policy.parameters(), grads):
        p.data -= 0.1 * g

test_traj = collect_trajectory(env, adapted_policy, steps=100, device=device)
rewards = sum(r for _, _, r, _ in test_traj)
print(f"Test reward after adaptation: {rewards}")

