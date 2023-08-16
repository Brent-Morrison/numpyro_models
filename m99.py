"""
# Testing dummy data for autoencoder
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d, e = dummy_data(n=13, scale=True, complex=False)
e_np = e.detach().numpy()

pca = PCA(n_components=3)

e_scores = pca.fit(e_np).transform(e_np)

print("Scaled e\n",np.round(e_np,3),"\n")
print("Components e\n",np.round(e_scores,3),"\n")

print("Raw d\n",d,"\n")
d_scaled = StandardScaler().fit_transform(d)
print("Scaled d\n",d_scaled,"\n")
d_scores = pca.fit(d_scaled).transform(d_scaled)
print("Components d\n",np.round(d_scores, 3),"\n")

print(np.sqrt(((e_scores - d_scores) ** 2).mean())) 

#layer_widths = [9,7]#,5,3,1]
#layer_widths = [9,7,5]#,3,1]
#layer_widths = [9,7,5,3]#,1]
layer_widths = [9,7,5,3,1]
encz = zip(layer_widths[:-1], layer_widths[1:])
if len(layer_widths) < 4:
    *hidden, last = encz
    for n_in, n_out in hidden:
        print(n_in, " - ", n_out, " Linear & Sigmoid")
    print(last[0], " - ", last[1], " Linear")
else:
    *hidden, second_last, last = encz
    for n_in, n_out in hidden:
        print(n_in, " - ", n_out, " Linear & Relu")
    print(second_last[0], " - ", second_last[1], " Linear & Sigmoid")
    print(last[0], " - ", last[1], " Linear")





# Print device
e1=1
e2=100
status1 = "Improved validation loss"
status2 = "1 epochs without improved validation loss"
print(f"Epoch {e1+1:3} | {status1:50} | ")
print(f"Epoch {e2+1:3} | {status2:50} | \n")





# Visualising train and validation error
error_df = pd.DataFrame({
    'Epoch' : list(range(1, len(train_error)+1)), 
    'Train' : train_error, 
    'Test': test_error})

# Plot 1
sns.set_style('darkgrid')
sns.lineplot(
    data=error_df.melt(id_vars=['Epoch'], value_vars=['Train', 'Test'], 
                        value_name='Error', var_name='Dataset'),
    x='Epoch',
    y='Error',
    hue='Dataset',
    palette=sns.color_palette("mako_r", 2)
    )

# Plot 2 #
sns.set_style('darkgrid')
g = sns.lineplot(data=error_df, x='Epoch', y='Train', label='Train')
sns.set_style('darkgrid', {'axes.grid': False})
sns.lineplot(data=error_df, x='Epoch', y='Test', label='Test', color='red', markers=False, ax=g.twinx()) \
    .set(title='Training and validation errror')


# Linear exponential loss
def linex(y, yhat, alpha = 0.5):
    error = y - yhat
    return torch.exp(alpha * torch.sign(y) * error) - (alpha * torch.sign(y)) * error - 1

# Test
#y = torch.Tensor([[0.5]]).repeat(7,1)
y1    = torch.Tensor([[-1.5],[-1],[-0.5],[0],[0.5],[1],[1.5]])
yhat1 = torch.Tensor([[-1.5],[-1],[-0.5],[0],[0.5],[1],[1.5]])

y1    = torch.Tensor([-1.5,-1,-0.5,0,0.5,1,1.5])
yhat1 = torch.Tensor([-1.5,-1,-0.5,0,0.5,1,1.5])

y, yhat = torch.meshgrid(y1, yhat1)
error = linex(y, yhat)

fig = plt.figure()
ax = plt.axes(projection='3d')


##############################

def linex(y, yhat, alpha = 0.5):
    error = y - yhat
    return np.exp(alpha * np.sign(y) * error) - (alpha * np.sign(y)) * error - 1

ys = torch.linspace(-1.5, 1.5, 30) #Tensor([-1.5,-1,-0.5,0,0.5,1,1.5])
yhats = torch.linspace(-1.5, 1.5, 30) #Tensor([-1.5,-1,-0.5,0,0.5,1,1.5])
y, yhat = torch.meshgrid(ys, yhats) #, indexing='xy')
z = linex(y, yhat)
#z = (yhat-y)**2
ax = plt.axes(projection='3d')
ax.plot_surface(y.numpy(), yhat.numpy(), z.numpy(), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('y')
ax.set_ylabel('yhat')
ax.set_zlabel('loss')
#ax.view_init(elev=30, azim=-60) # default, elev: angle above/below x-y, azim: rotate about the z axis
ax.view_init(10, -15)
plt.show()



##############################

import json

with open("/c/Users/brent/Documents/R/Misc_scripts/e01/01-scripts_02-args.json", "r") as read_file:
    args = json.load(read_file)

l = args['predictors']
"""

# https://gist.github.com/matthewfeickert/3b7d30e408fe4002aac728fc911ced35
import argparse
import json

def main(args):
    for i in range(2):
        print(args.loss[i], " | ", args.layer_width[i], " | ", args.xcols)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(
        description='configuration arguments provided at run time from the CLI'
    )
    cli_parser.add_argument(
        '-c',
        '--config_file',
        dest='config_file',
        type=str,
        default=None,
        help='config file',
    )

    args, unknown = cli_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[cli_parser], add_help=False)

    if args.config_file is not None:
        if '.json' in args.config_file:
            config = json.load(open(args.config_file))
            parser.set_defaults(**config)

            [
                parser.add_argument(arg)
                for arg in [arg for arg in unknown if arg.startswith('--')]
                if arg.split('--')[-1] in config
            ]

    args = parser.parse_args()

    main(args)



##############################
"""
sys.path.insert(1, '/c/Users/brent/Documents/VS_Code/postgres/postgres/python')  # to be replaced once package set up
import pg_connect as pgc
import json

# Load config file
with open('/c/Users/brent/Documents/VS_Code/postgres/postgres/config.json', 'r') as f:
    config = json.load(f)

# Connect to db
conn = pgc.pg_connect(pg_password=config['pg_password'], database='stock_master')






# Sliding window
for window_num, i in enumerate(range(start_month_idx, loops*test_months, test_months)):
    start_idx = i
    end_idx = i + sample_months
    window = list(range(start_idx, end_idx))
    print(window_num)
    print("Sliding window --", start_idx, "-", str(months[start_idx])[:10], ":", end_idx,"-", str(months[end_idx-1])[:10])

    # Train / test split
    print("Train -----------", start_idx, "-", str(months[start_idx])[:10], ":", i + train_months,"-", str(months[i + train_months-1])[:10])
    train = list(range(start_idx, i + train_months))
    
    ##### TEST MODEL HERE #####

    print("Test ------------", i + train_months, "-", str(months[i + train_months])[:10], ":", end_idx,"-", str(months[end_idx-1])[:10])
    test = list(range(i + train_months, end_idx))
    print("\n")
"""

# Also - https://stackoverflow.com/questions/29027792/get-average-value-from-list-of-dictionary
import pickle
from collections import OrderedDict
with open("m03_best_model_list", "rb") as f:   # Unpickling
    best_model_list = pickle.load(f)

wndw_mdl_state = best_model_list[0]

for key in best_model_list[0]:
    # TO DO - this needs to be dynamic for differing kfolds
    wndw_mdl_state[key] = (best_model_list[0][key] + best_model_list[1][key] + best_model_list[2][key]) / len(best_model_list)
wndw_mdl_state



with open("m03_best_model_list", "rb") as f:   # Unpickling
    best_model_list = pickle.load(f)

wndw_mdl_state = best_model_list[0]
wndw_mdl_state = dict.fromkeys(wndw_mdl_state, 0) # Copy structure setting values to nil
wndw_mdl_state = OrderedDict((k, 0) for k in wndw_mdl_state)

#for m in range(len(best_model_list)):
for key in wndw_mdl_state:
    for m in range(len(best_model_list)):
        wndw_mdl_state[key] += best_model_list[m][key]
    wndw_mdl_state[key] /= len(best_model_list)
wndw_mdl_state



l = [1,2,3,4,5]
p = 0.
for i in l:
    p += i
p /= len(l)
print(p)