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
