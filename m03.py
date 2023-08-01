from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler

torch.set_printoptions(linewidth=120)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Prepare data
file_path="/c/Users/brent/Documents/R/Misc_scripts/data_test33/2022-12-11.csv"
#file_path = "https://raw.githubusercontent.com/Brent-Morrison/Misc_scripts/master/stock_data.csv"
df_raw = pd.read_csv(file_path)
df_raw['date_stamp'] = pd.to_datetime(df_raw['date_stamp'], format="%d/%m/%Y")
df = df_raw[df_raw['date_stamp'] >= '2017-06-01'].reset_index(drop=True).copy()
ticker = ['symbol']
date = ['date_stamp']
ycols = ['fwd_rtn_1m']
xcols = ['rtn_ari_1m', 'rtn_ari_3m', 'rtn_ari_12m', 'vol_ari_60d', 'vol_ari_120d', 'skew_ari_120d', 'kurt_ari_120d']
df.sort_values(by=[ticker[0], date[0]], ascending=[True, True], inplace=True)
df['fwd_rtn_1m'] = df.groupby('symbol')['adjusted_close'].pct_change(periods=1).shift(periods=-1)
df = df.loc[:, date+ticker+ycols+xcols].copy()
df_oos = df[df['date_stamp'] >= max(df['date_stamp'])].reset_index(drop=True).copy()
df.dropna(inplace=True)



# --------------------------------------------------------------------------------------------------------------------------
# DF dataloader
# https://averdones.github.io/reading-tabular-data-with-pytorch-and-training-a-multilayer-perceptron/
# https://hutsons-hacks.info/building-a-pytorch-binary-classification-multi-layer-perceptron-from-the-ground-up
#
# --------------------------------------------------------------------------------------------------------------------------

class DFLoader(Dataset):
    """Loads a dataframe"""

    def __init__(self, df, date, ycols, xcols):
        """Initializes instance of class DFLoader.

        Args:
            file_path (str): Path to the csv file required
            date(str): date column
        """
        
        # Read CSV file
        #df = pd.read_csv(file_path, usecols=date+ycols+xcols)  # include stock ticker
        #df['date_stamp'] = pd.to_datetime(df['date_stamp'], format="%d/%m/%Y")
        #df = df[df['date_stamp'] >= date_filter].reset_index(drop=True).copy()
        self.df = df

        # Dependent variable ("y") and predictors ("x")
        self.x = torch.from_numpy(df.loc[:, xcols].values)
        self.y = torch.from_numpy(df.loc[:, ycols].values)
        #self.d = df.loc[:, date].values
        self.d = df[date[0]]
        self.d = pd.to_datetime(self.d, format="%d/%m/%Y")
        self.d = self.d.to_numpy()
        self.x = self.x.to(torch.float32)
        self.y = self.y.to(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


dataset = DFLoader(df, date, ycols, xcols)




# --------------------------------------------------------------------------------------------------------------------------
# 
# Neural network
#
# TO DO : Custom loss function
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3973086
# https://www.sciencedirect.com/science/article/abs/pii/S0304407606001606?via%3Dihub
# 
# --------------------------------------------------------------------------------------------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self, xcols):
        super(NeuralNetwork, self).__init__()
        features = len(xcols)
        hl1_dim = max(round(len(xcols)/2), 2)
        self.linear_relu_stack = nn.Sequential(
            # out_features = number of nodes
            nn.Linear(in_features=features, out_features=features), 
            nn.ReLU(),
            nn.Linear(in_features=features, out_features=hl1_dim),
            nn.ReLU(),
            nn.Linear(in_features=hl1_dim, out_features=1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# Test
model = NeuralNetwork(xcols=xcols).to(device)
print(model)

# Hyperparameters
learning_rate = 1e-2
batch_size = 50

# Initialize the loss function
loss_fn = nn.MSELoss()

# Optimiser
optimizer = optim.SGD(model.parameters(), lr=learning_rate)




# Define train loop
def train_loop(dataloader, model, loss_fn, optimizer, verbose=False):
    size = len(dataloader.sampler)
    running_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Running loss
        running_loss += loss.item()

        # Print loss
        if batch % 20 == 0 and verbose == True:
            loss = loss.item()
            current = batch * len(X)
            print(f"Training error: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return running_loss / size



# Define validation / test loop
def test_loop(dataloader, model, loss_fn, verbose=False):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches # same as 'tl = tl / nb'

    if verbose == True:
        print(f"\nTest error: {test_loss:>7f} [{len(X):>5d}] \n")
    
    return test_loss  




# Early stopping
class earlyStopping():
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_vldtn_error = np.inf
        self.status = ""

    def __call__(self, vldtn_error):
        if vldtn_error < self.min_vldtn_error:
            self.min_vldtn_error = vldtn_error
            self.counter = 0
            self.status = f"Improved validation loss\n"
        elif vldtn_error > (self.min_vldtn_error + self.min_delta):
            self.counter += 1
            self.status = f"{self.counter} epochs without improved validation loss\n"
            if self.counter >= self.patience:
                self.status = f"{self.counter} epochs without improvement, patience exceeded\n"
                self.early_stop = True

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.min_vldtn_error = np.inf
        self.status = ""



# Early stopping test
"""for i in range(len(test_error)):
    error = test_error[i]
    es(error)
    print(f"Error: {error:>5f} |", es.status)
    if es.early_stop:
        break
"""





# Train / test loop
def train_test_loop(train_loader, test_loader, model, loss_fn, optimizer, epochs, early_stop, weights=None, verbose=False):
    train_error = []
    test_error = []
    best_model_state = []
    start_state = []

    torch.manual_seed(42)

    # Apply weight initialisation
    # https://discuss.pytorch.org/t/reset-model-weights/19180
    # https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            m.bias.data.fill_(0.01)
    
    if weights is None:
        model.apply(weights_init)
    else:
        model.load_state_dict(weights)
    
    start_state = deepcopy(model.state_dict())

    es = early_stop
    
    for e in range(epochs):
        print(f"Epoch {e+1} ---------------------------------------------\n")
        train_error.append(train_loop(train_loader, model, loss_fn, optimizer, verbose))
        test_error.append(test_loop(test_loader, model, loss_fn, verbose))
        es(test_error[e])
        print(f"Error: {test_error[e]:>5f} |", es.status)
        if e+1 == epochs:
            # Save model on completion of epochs and reset early stop
            best_model_state = deepcopy(model.state_dict())
            es.reset()
        elif es.early_stop:
            # Save model on early stop and reset early stop
            best_model_state = deepcopy(model.state_dict())
            es.reset()
            break
    
    print("Complete\n")
    
    return train_error, test_error, best_model_state, start_state






# Train / test parameters
train_months = 30
test_months = 6
months = np.sort(np.unique(dataset.d))
n_months = len(np.unique(dataset.d))
sample_months = train_months + test_months 
loops = int(np.floor((n_months - train_months) / test_months)) 
start_month_idx = int(n_months - (test_months * loops) - train_months)
es = earlyStopping(patience=15)

# Test sliding window loop ------------------------
# K-fold 
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Date index to record index
idx_date_dict = {k: v for k, v in enumerate(list(dataset.d.reshape(-1)))}
oos_pred_list = []
wndw_mdl_states = []
wndw_mdl_state = None
start_states = []
start_state = None
epochs = 10

# Sliding window
for window_num, i in enumerate(range(start_month_idx, loops*test_months, test_months)):
    start_idx = i
    end_idx = i + sample_months
    window = list(range(start_idx, end_idx))
    
    print("Sliding window --", start_idx, "-", str(months[start_idx])[:10], ":", end_idx,"-", str(months[end_idx-1])[:10])

    # Train / test split
    print("Train -----------", start_idx, "-", str(months[start_idx])[:10], ":", i + train_months,"-", str(months[i + train_months-1])[:10])
    train = list(range(start_idx, i + train_months))
    
    best_model_list = []
    # K fold cross validation
    for fold, (train_sub, validate) in enumerate(kf.split(np.array(train))): 
        print("CV Fold:", fold + 1, "%s %s" % (train_sub, validate))
        
        # Propagate months to individual records - training
        train_dict = {k: v for (k, v) in idx_date_dict.items() if v in list(months[train_sub])}
        train_idx = list(train_dict.keys())
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        
        # Propagate months to individual records - validation
        vldtn_dict = {k: v for (k, v) in idx_date_dict.items() if v in list(months[validate])}
        vldtn_idx = list(vldtn_dict.keys())
        vldtn_sampler = SubsetRandomSampler(vldtn_idx)
        vldtn_loader = DataLoader(dataset, sampler=vldtn_sampler, batch_size=len(vldtn_idx)) # 1 batch for validation

        # Train & validate
        train_error, test_error, best_model_state, start_state = train_test_loop(
            train_loader, vldtn_loader, model, loss_fn, optimizer, epochs, 
            early_stop=es, 
            weights=None if window_num == 0 else wndw_mdl_state
            )
        best_model_list.append(best_model_state)

    
    ##### TEST MODEL HERE #####

    print("Test ------------", i + train_months, "-", str(months[i + train_months])[:10], ":", end_idx,"-", str(months[end_idx-1])[:10])
    test = list(range(i + train_months, end_idx))
    print("\n")
    
    # Create average of CV folds
    wndw_mdl_state = best_model_list[0]
    for key in best_model_list[0]:
        # this needs to be dynamic for differing kfolds
        wndw_mdl_state[key] = (best_model_list[0][key] + best_model_list[1][key] + best_model_list[2][key]) / len(best_model_list)
    
    # Track model states
    wndw_mdl_states.append(wndw_mdl_state)
    start_states.append(start_state)

    # Load state dict
    model.load_state_dict(wndw_mdl_state)
    model.eval()

    # Propagate months to individual records - test
    test_dict = {k: v for (k, v) in idx_date_dict.items() if v in list(months[test])}
    test_idx = list(test_dict.keys())
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Apply to new data
    with torch.no_grad():
        y_hat = []
        y_true = []
        for X, y in test_loader:
            y_hat.append(model(X).item())
            y_true.append(y.item())

    # Write to data frame and insert into list
    oos_pred_temp_df = df[df['date_stamp'].isin(months[test])][['date_stamp', 'symbol', 'fwd_rtn_1m']] \
        .reset_index(drop=True).copy()
    oos_pred_temp_df['y_true'] = y_true
    oos_pred_temp_df['y_hat'] = y_hat

    """oos_pred_temp_df = pd.DataFrame({
        'y_hat' : y_hat, 
        'y_true':  y_true
        })"""
    
    oos_pred_list.append(oos_pred_temp_df)
    
    
# Compute performance metrics
oos_pred_df = pd.concat(oos_pred_list)








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
y1 = torch.Tensor([[-1.5],[-1],[-0.5],[0],[0.5],[1],[1.5]])
yhat1 = torch.Tensor([[-1.5],[-1],[-0.5],[0],[0.5],[1],[1.5]])

y1 = torch.Tensor([-1.5,-1,-0.5,0,0.5,1,1.5])
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
