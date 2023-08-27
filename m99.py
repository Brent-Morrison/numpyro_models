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


# Synthetic stock data
import numpy as np
import pandas as pd
import datetime as dt
import random as r
from string import ascii_lowercase
import matplotlib.pyplot as plot

def gen_stock_data(n_sectors=4, n_stocks=5, months=36):
    sect = [chr(x) for x in range(65, 65 + n_sectors)]
    sect = [item for item in sect for _ in range(n_stocks)]
    stocks = [''.join([r.choice(ascii_lowercase) for _ in range(3)]) for _ in range(n_sectors*n_stocks)]  
    df = pd.DataFrame({"sector" : sect, "stock" : stocks})
    dates = pd.date_range(dt.datetime(2020,1,1), periods=months, freq="M").tolist()
    df = pd.merge(df, pd.DataFrame({"date_stamp" : dates, "mkt_state" : np.sin(np.arange(months)/(months/10))}), how="cross")
    # Market specific component to stock return depends on mkt_state
    df["mkt_rtn_beta"] = np.where(df["mkt_state"] < 0, -0.1, 0.2) 
    df["mkt_mean_rtn"] = df["mkt_rtn_beta"] * df["mkt_state"]
    df["mkt_stdev_rtn"] = np.where(df["mkt_state"] < 0, 0.05, 0.02) 
    df["mkt_rtn"] = np.random.normal(loc=df["mkt_mean_rtn"], scale=df["mkt_stdev_rtn"])
    # Sector specific component to stock return depends on mkt_state and sect_state  
    df["sect_state"] = np.linspace(0.1, 0.2, n_sectors).repeat(months*n_stocks)
    df["sect_rtn_beta"] = np.where((df["mkt_state"] < 0) & (df["sect_state"] <= 0.5), 0.1, -0.1) 
    df["sect_mean_rtn"] = df["sect_rtn_beta"] * df["sect_state"]
    df["sect_stdev_rtn"] = np.where(df["sect_state"] <= 0.1, 0.02, 0.04) 
    df["sect_rtn"] = np.random.normal(loc=df["sect_mean_rtn"], scale=df["sect_stdev_rtn"])
    # Stock specific component to stock return depends on mkt_state and sect_state and stock_state
    df["stock_state"] = np.where(df["date_stamp"] < dates[int(months/2)], 0.1, 0.2) 
    df["stock_state1"] = np.where(df["date_stamp"] < dates[int(months/3)], 0.1, 0.2) 
    df["stock_rtn_beta"] = np.select(
        [(df["mkt_state"] < 0  ) & (df["sect_state"] == 0.2),
         (df["mkt_state"] < 0  ) & (df["sect_state"] == 0.1),
         (df["mkt_state"] > 0.5) & (df["sect_state"] >= 0  )],
         [0.05, 
          0.02,
          -0.02],
          default=0)
    df["stock_mean_rtn"] = df["stock_rtn_beta"] * df["stock_state"]
    df["stock_stdev_rtn"] = np.where(df["stock_state"] <= 0.1, 0.02, 0.04) 
    df["stock_rtn"] = np.random.normal(loc=df["stock_mean_rtn"], scale=df["stock_stdev_rtn"])

    # Stock return
    df["stock_rtn"] = df["mkt_rtn"] + df["sect_rtn"] + df["stock_rtn"]
    df["stock_rtn_binary"] = np.where(df["stock_rtn"] < 0, 0, 1)


    return df

df = gen_stock_data()


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