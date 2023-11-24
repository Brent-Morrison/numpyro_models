"""

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

"""
# ---------------------------------------------------------------------------------------------------------

hist = {"train_loss" : [], "vldtn_loss" : [], "test_loss" : []}

for i in range(9):
    hist["train_loss"].append(i)
    hist["vldtn_loss"].append(i+1)
    hist["test_loss"].append(i+2)

# python m99.py -c m03_config.json
# ./m03_run.sh

# ---------------------------------------------------------------------------------------------------------

def expand_grid(layer_width=[[5,1]], epochs=[20], es_patience=[10], learning_rate=[1e-2,1e-3], loss_fn=["linex"]):
    lgrid = [[x, y, z, a, b] \
        for x in layer_width   \
        for y in epochs        \
        for z in es_patience   \
        for a in learning_rate \
        for b in loss_fn       ]
    dgrid = {"layer_width"   : [l[0] for l in lgrid],
             "epochs"        : [l[1] for l in lgrid],
             "es_patience"   : [l[2] for l in lgrid],
             "learning_rate" : [l[3] for l in lgrid],
             "loss_fn"       : [l[4] for l in lgrid]}
    return dgrid

expand_grid(loss_fn=["linex", "mse"])
expand_grid(loss_fn=["linex", "mse"])["layer_width"][0]


# ---------------------------------------------------------------------------------------------------------
# Synthetic stock data (m04.py)

df = gen_stock_data()[["date_stamp", "mkt_state", "sector", "stock", "stock_state0", "stock_rtn"]]

X, Y, idx = get_data(filepath="/c/Users/brent/Documents/R/Misc_scripts/stocks.csv", date_filter="2022-12-31")




months=36
dates =  pd.date_range(dt.datetime(2020,1,1), periods=months, freq="M").tolist()
time = np.arange(months)/(months/10)
amplitude = np.sin(time)
plot.plot(time, amplitude)
plot.show()

np.random.normal(loc=np.array([3,3,3,6,6,6]), scale=np.array([0.1,0.1,0.1,1,1,1]))











def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


df = expand_grid(
    {"height": [60, 70], "weight": [100, 140, 180], "sex": ["Male", "Female"]}
)

grid = [[chr(x),chr(y)] \
        for x in range(65, 65 + sectors) \
        for y in range(97, 97 + stocks)]
grid

pd.date_range(dt.datetime(2020,1,1), periods=10, freq="M").tolist()
[item for item in sect for _ in range(sectors)]




def gen_polynomial(B, n):
    x = np.random.normal(size=10)
    y = B.dot

x = [1, .5]
x1 = np.array(x)[:,None].T
x2 = np.random.normal(size=10)[:,None]
y = x2.dot(x1)



# https://github.com/MaxRobinsonTheGreat/mandelbrotnn/blob/main/src/models.py
import numpy as np
xmin=-2.5
xmax=1.0
ymin=-1.1
ymax=1.1
x_size=10
y_size=10

if x_size is not None:
    x_m = x_size/(xmax - xmin)
else: 
    x_m = 1.
if y_size is not None:
    y_m = y_size/(ymax - ymin)
else: 
    y_m = 1.
x_b = -(xmin + xmax)*x_m/2 - 1 
y_b = -(ymin + ymax)*y_m/2
m = np.array([x_m, y_m])
b = np.array([x_b, y_b])

# Data
x = np.random.normal(size=10)[:,None]
m*x + b