import pickle, torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from cheatools.lgnn import lGNN
from copy import deepcopy

for s in ['train','val']:
    with open(f'C:/Users/Tseh/Documents/Files/HEA/hea_project/train_data_graphs/graphs/{s}.graphs', 'rb') as input:
        globals()[f'{s}_graphs'] = pickle.load(input)

filename = 'lGNN'
torch.manual_seed(42)

batch_size = 32
max_epochs = 1000
learning_rate = 5e-4

roll_val_width = 30
patience = 150
report_every = 25

input_dim = train_graphs[0].x.shape[1]

arch = {'n_conv_layers': 2,
        'n_hidden_layers': 1,
        'conv_dim': 32,
        'input_dim': input_dim,
        'act': 'relu',
       }

model = lGNN(arch=arch)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=50, verbose=True)

train_loader = DataLoader(train_graphs, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), drop_last=True, shuffle=False)

train_loss, val_loss = [], []
model_states = []

for epoch in range(max_epochs):
    train_loss.append(model.train4epoch(train_loader, batch_size, opt))
    pred, target, _ = model.test(val_loader, len(val_graphs))
    val_mae = abs(np.array(pred) - np.array(target)).mean()
    val_loss.append(val_mae)
    model_states.append(deepcopy(model.state_dict()))
    
    scheduler.step(val_mae)

    if epoch >= roll_val_width+patience:
        roll_val = np.convolve(val_loss, np.ones(int(roll_val_width+1)), 'valid') / int(roll_val_width+1)
        min_roll_val = np.min(roll_val[:-patience+1])
        improv = (roll_val[-1] - min_roll_val) / min_roll_val

        if improv > -0.005:
            print('Early stopping invoked.')
            best_epoch = np.argmin(val_loss)
            best_state = model_states[best_epoch]
            break

    if epoch % report_every == 0:
        print(f'Epoch {epoch} train and val L1Loss: {train_loss[-1]:.3f} / {val_loss[-1]:.3f} eV')

print(f'Finished training sequence. Best epoch was {best_epoch} with val. L1Loss {np.min(val_loss):.3f} eV')

best_state['onehot_labels'] = train_graphs[0].onehot_labels
best_state['arch'] = arch

with open(f'{filename}.state', 'wb') as output:
    pickle.dump(best_state, output)

fig, main_ax = plt.subplots(1, 1, figsize=(8, 5))
color = ['steelblue','green']
label = [r'Training set  L1Loss',r'Validation set L1Loss']

for i, results in enumerate([train_loss, val_loss]):
    main_ax.plot(range(len(results)), results, color=color[i], label=label[i])
    if i == 1:
        main_ax.scatter(best_epoch, val_loss[best_epoch], facecolors='none', edgecolors='maroon', label='Best epoch', s=50, zorder=10)

main_ax.set_xlabel(r'Epoch', fontsize=16)
main_ax.set_ylabel(r'L1Loss [eV]', fontsize=16)
main_ax.legend()

plt.savefig(f'{filename}_curve.png')
