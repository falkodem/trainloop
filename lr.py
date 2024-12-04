import matplotlib.pyplot as plt
import torch
epochs = 100
steps_per_epoch = 200

model = torch.nn.Linear(100, 10)

optimizer = torch.optim.AdamW(model.parameters(),
                        lr=0.01,
                        betas=(0.9, 0.999),
                        weight_decay=0.01)

def func(epoch):
    return (1/(epoch/100+1))*(torch.cos(torch.tensor(epoch/5)) + 1.01).item()

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
# lin_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.7, total_iters=5)
# exp_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
# cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.001, T_max=10,)
# # exp_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0.001, T_0=20, T_mult=1)
# scheduler = torch.optim.lr_scheduler.ChainedScheduler([cos_sched, exp_sched], optimizer)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [lin_sched, chained], milestones=[5])

learning_rates1 = []
for epoch in range(1, epochs + 1):
    for iter in range(steps_per_epoch):
        lr = optimizer.param_groups[0]['lr']
        learning_rates1.append(lr)
        optimizer.step()
    scheduler.step()
iterations = list(range(1, len(learning_rates1)+1))
plt.plot(iterations, learning_rates1)
plt.savefig('LR.png')