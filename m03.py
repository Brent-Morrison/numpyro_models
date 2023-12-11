import argparse
import json
from copy import deepcopy
from collections import OrderedDict
import io
from google.cloud import storage # TO DO - add this to env
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler


# Convenience functions ----------------------------------------------------------------------------------------------------

# https://www.geeksforgeeks.org/how-to-pass-a-list-as-a-command-line-argument-with-argparse/
def list_of_strings(strings):
    return strings.split(',')

def list_of_ints(ints):
    return list(map(int, ints.split(',')))
    
def expand_grid0(*args):
    if len(args) == 2:
        grid = [[x, y] for x in args[0] for y in args[1]]
    elif len(args) == 3:
        grid = [[x, y, z] for x in args[0] for y in args[1] for z in args[2]]
    return grid


def expand_grid(layer_width=[[5,1]], epochs=[20], es_patience=[10], learning_rate=[1e-2,1e-3], loss_fn=["linex"]):
    grid = [[x, y, z, a, b] \
        for x in layer_width   \
        for y in epochs        \
        for z in es_patience   \
        for a in learning_rate \
        for b in loss_fn       ]
    return grid


# GCP functions

def gcp_csv_to_df(bucket_name, source_file_name):
    """
    Grab CSV file from GCP Storage and return as DF
    File extension is NOT required for parameter "source_file_name" (TO DO - check this)
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    return df

def write_df_gcs_csv(df, bucket_name, blob_name):
    """
    Use pandas to interact with GCS using file-like IO
    File extension IS required for parameter "blob_name"
    The ID of your GCS bucket: bucket_name = "your-bucket-name"
    The ID of your new GCS object: blob_name = "storage-object-name"
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with blob.open("w") as f:
        f.write(df.to_csv(index=False))

    print(f"Wrote csv with pandas with name {blob_name} from bucket {bucket_name}.")


# Linear exponential loss --------------------------------------------------------------------------------------------------

class LinexLoss(nn.Module):

    def __init__(self, alpha=0.5):
        super(LinexLoss, self).__init__()
        self.alpha = alpha

    def forward(self, yhat, y):
        error = y - yhat
        linex = torch.exp(self.alpha * torch.sign(y) * error) - (self.alpha * torch.sign(y)) * error - 1
        return torch.mean(linex)
    
    
    

# Get data -----------------------------------------------------------------------------------------------------------------

def get_data(file_path, date_filter, ycols, xcols, bucket_name=None, source_file_name=None): # TO DO - toggle getting data from GCP storage and locally
    """Load a data frame from csv file. Assumes presence of 'fwd_rtn' and 'date_stamp' columns

    Args:
        file_path (str)  : path to csv file
        date_filter (str): remove dates before this date
    """
    #file_path = "/c/Users/brent/Documents/R/Misc_scripts/e01/02-data_01-training.csv"
    #file_path = "https://raw.githubusercontent.com/Brent-Morrison/Misc_scripts/master/stock_data.csv"
    if bucket_name is None:
        df_raw = pd.read_csv(file_path)
    else:
        df_raw = gcp_csv_to_df(bucket_name, source_file_name)
    df_raw['date_stamp'] = pd.to_datetime(df_raw['date_stamp'])#, format="%Y/%m/%d")
    df = df_raw[df_raw['date_stamp'] >= date_filter].reset_index(drop=True).copy()
    ticker = ['symbol']
    date = ['date_stamp']
    #ycols = ['fwd_rtn']
    #xcols = ['rtn_ari_1m', 'rtn_ari_3m', 'rtn_ari_12m', 'perc_range_12m', 'perc_pos_12m', 'rtn_from_high_12m', 'vol_ari_60d']
    df.sort_values(by=[ticker[0], date[0]], ascending=[True, True], inplace=True)
    df = df.loc[:, date+ticker+ycols+xcols].copy()
    df.dropna(inplace=True)
    
    return df




# DF dataloader ------------------------------------------------------------------------------------------------------------

class DFLoader(Dataset):
    """Loads a dataframe"""

    def __init__(self, df, date, ycols, xcols):
        """Initializes instance of class DFLoader.

        Args:
            df (str)     : Data frame to load
            date (str)   : date column
            ycols (list) : target
            xcols (list) : predictors
        """
        
        # Dataframe
        self.df = df

        # Dependent variable ("y") and predictors ("x")
        self.x = torch.from_numpy(df.loc[:, xcols].values)
        self.y = torch.from_numpy(df.loc[:, ycols].values)
        self.d = df['date_stamp']
        self.d = pd.to_datetime(self.d, format="%d/%m/%Y")
        self.d = self.d.to_numpy()
        self.x = self.x.to(torch.float32)
        self.y = self.y.to(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]





# --------------------------------------------------------------------------------------------------------------------------
# 
# Neural network
# 
# --------------------------------------------------------------------------------------------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self, layer_widths):
        super(NeuralNetwork, self).__init__()
        self.network = nn.ModuleList()
        zipped = zip(layer_widths[:-1], layer_widths[1:])
        if len(layer_widths) < 4:
            *hidden, last = zipped
            for n_in, n_out in hidden:
                self.network.append(nn.Linear(n_in, n_out))
                self.network.append(nn.Sigmoid())
            self.network.append(nn.Linear(last[0], last[1]))
        else:
            *hidden, second_last, last = zipped
            for n_in, n_out in hidden:
                self.network.append(nn.Linear(n_in, n_out))
                self.network.append(nn.ReLU())
            self.network.append(nn.Linear(second_last[0], second_last[1]))
            self.network.append(nn.Sigmoid())
            self.network.append(nn.Linear(last[0], last[1]))           
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return self.network(x)




# Define train loop
def train_loop(dataloader, model, loss_fn, optimizer, verbose=False):
    """Completes one epoch looping over all batches
    
    Returns average error over epoch
    """
    size = len(dataloader.sampler)
    running_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        
        # Compute prediction and loss
        yhat = model(X)
        loss = loss_fn(yhat, y)

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
    """Completes one model looping over all batches (batches typically set to one for test)
    
    Returns average error
    """
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            yhat = model(X)
            test_loss += loss_fn(yhat, y).item()
    
    test_loss /= num_batches # same as 'tl = tl / nb'

    if verbose == True:
        print(f"Error: {test_loss:>4f}")
    
    return test_loss  




# Early stopping
class earlyStopping():
    """TO DO
    
    xxxxxx
    """
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
            self.status = f"Improved validation loss"
        elif vldtn_error > (self.min_vldtn_error + self.min_delta):
            self.counter += 1
            self.status = f"{self.counter} epochs without improved validation loss"
            if self.counter >= self.patience:
                self.status = f"{self.counter} epochs without improvement, patience exceeded"
                self.early_stop = True

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.min_vldtn_error = np.inf
        self.status = ""





# Train / test loop
def train_test_loop(train_loader, test_loader, model, loss_fn, optimizer, epochs, early_stop, weights=None, verbose=False):
    
    """Completes one model assessment over all samples
    
    Returns:
        train_error (list)      : each epochs training error
        test_error (list)       : each epochs test error 
        best_model_state (dict) : state_dict as at last epoch
        start_state (dict)      : the state_dict loaded or initialised 
    """
    
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
        train_error.append(train_loop(train_loader, model, loss_fn, optimizer, verbose))
        test_error.append(test_loop(test_loader, model, loss_fn, verbose))
        es(test_error[e])
        print(f"Epoch {e+1:3} | Error: {test_error[e]:>4f} | {es.status:43}")
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





# Main ---------------------------------------------------------------------------------------------------------------------

def main(args):
    
    torch.set_printoptions(linewidth=120)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device\n")
    
    # Data
    ycols = ['fwd_rtn']
    xcols = args.xcols
    df = get_data(
        file_path=args.source_file, 
        date_filter="2016-12-31", 
        xcols=xcols, 
        ycols=ycols,
        bucket_name=None if args.bucket_name == "None" else args.bucket_name,
        source_file_name=None if args.source_file_name == "None" else args.source_file_name
        )
    date = 'date_stamp'
    dataset = DFLoader(df, date, ycols, xcols)

    # Train / test parameters
    train_months = 12
    test_months = 6
    months = np.sort(np.unique(dataset.d))
    n_months = len(np.unique(dataset.d))
    sample_months = train_months + test_months 
    loops = int(np.floor((n_months - train_months) / test_months)) 
    start_month_idx = int(n_months - (test_months * loops) - train_months)

    # K-fold 
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Date index to record index
    idx_date_dict = {k: v for k, v in enumerate(list(dataset.d.reshape(-1)))}
    oos_pred_list = []
    best_model_window = None
    start_states = []
    start_state = None

    # Sliding window (start) --------------------------------------------------------------------------------------------(3)
    for window_num, i in enumerate(range(start_month_idx, loops*test_months, test_months)):
        
        start_idx = i
        end_idx = i + sample_months
        window = list(range(start_idx, end_idx))
        train = list(range(start_idx, i + train_months))
        vldtn_error_param = []
        best_model_param = []
        
        grid = expand_grid( \
            layer_width=args.layer_width, epochs=args.epochs, \
            es_patience=args.es_patience, learning_rate=args.learning_rate, \
            loss_fn=args.loss_fn)
        
        # Loop over hyper parameters (start) ----------------------------------------------------------------------------(2)
        for p in range(len(grid)):
            print("------------------------------------------------------")
            print(f"Window {window_num+1:2}\n")
            print(f"Hyper-parameter set {p+1} of {len(grid)}")
            rj=25
            print("-Epochs:".ljust(rj),str(grid[p][1]).rjust(rj))
            print("-Early stop patience:".ljust(rj),str(grid[p][2]).rjust(rj))
            print("-Learning rate:".ljust(rj),str(grid[p][3]).rjust(rj))
            print("-Loss function:".ljust(rj),str(grid[p][4]).rjust(rj),"\n")
            

            # Model
            layer_widths = grid[p][0]  #[5,1]  #args.layer_width
            layer_widths = [len(xcols)] + layer_widths
            model = NeuralNetwork(layer_widths).to(device)
            epochs = grid[p][1] #5 #args.epochs
            print(model,"\n")
            print("------------------------------------------------------\n")
            
            batch_size = 10
            
            # Initialize the loss function & optimiser
            if grid[p][4] == 'mse':
                loss_fn = nn.MSELoss()
            elif grid[p][4] == 'mae':
                loss_fn = nn.L1Loss()
            elif grid[p][4] == 'linex':
                loss_fn = LinexLoss()
            optimizer = optim.SGD(model.parameters(), lr=grid[p][3])
            es = earlyStopping(patience=grid[p][2])
            
            best_model_fold = []
            vldtn_error_fold = []
            
            # Training - K fold cross validation (start) --------------------------------------------------------------- (1)
            for fold, (train_sub, validate) in enumerate(kf.split(np.array(train))): 
                print(f"Window {window_num+1:2} | Train {str(months[start_idx])[:10]} to {str(months[i + train_months-1])[:10]} | CV Fold {fold + 1}")
                print("------------------------------------------------------")
                
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
                train_error, vldtn_error, best_model_state, start_state = train_test_loop(
                    train_loader, vldtn_loader, model, loss_fn, optimizer, epochs, 
                    early_stop=es, 
                    # Only use the prior window model parameters if hyper-paramater grid is not testing different model layer configurations
                    weights=None if len(args.layer_width) != 1 else best_model_window #TO DO - CHECK WHEN THIS SHOULD BE NONE SO IS NOT CONTAMINATED BY PRIOR FOLDS, EACH FOLD SHOULD HAVE A COLD START?
                    )
                
                vldtn_error_fold.append(vldtn_error[-1:][0])  # list of val error for each fold
                best_model_fold.append(best_model_state)      # list of model for each fold
            
            # Training - K fold cross validation (end) ------------------------------------------------------------------(1)
            
            
            # Average error per parameter setting (over all CV folds)
            vldtn_error_param_ = np.mean(np.array(vldtn_error_fold))
            vldtn_error_param.append(vldtn_error_param_)
            print("Validation error", np.round(vldtn_error_param_, 6),"\n")
            
            # Average of model state per parameter setting (over all CV folds)
            best_model_param_ = best_model_fold[0]
            best_model_param_ = OrderedDict((k, 0) for k in best_model_param_) # Copy structure setting values to nil, "dict.fromkeys(best_model_param_, 0)" coverts to regular dict
            for key in best_model_param_:
                for m in range(len(best_model_fold)):
                    best_model_param_[key] += best_model_fold[m][key]
                best_model_param_[key] /= len(best_model_fold)
            best_model_param.append(best_model_param_)
        
        # Loop over hyper parameters (end) ------------------------------------------------------------------------------(2)


        best_hyper_param_idx = vldtn_error_param.index(min(vldtn_error_param))
        print("Best hypers", grid[best_hyper_param_idx],"\n")
        
        # Testing
        print(f"Window {window_num+1:2} | Test  {str(months[i + train_months])[:10]} to {str(months[end_idx-1])[:10]}")
        print("------------------------------------------")
        test = list(range(i + train_months, end_idx))
        
        # Track model states
        best_model_window = best_model_param[best_hyper_param_idx]
        start_states.append(start_state)
        
        # Instantiate test model (required if layer_width differs across best model and that per last hyper parameter loop)
        layer_widths = grid[best_hyper_param_idx][0]
        layer_widths = [len(xcols)] + layer_widths
        model = NeuralNetwork(layer_widths).to(device)
        
        # Load state dict
        model.load_state_dict(best_model_window)
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
        
        # Error on test data
        test_loop(test_loader, model, loss_fn, verbose=True)
        print("\n")

        # Write to temp data frame and append df into list
        oos_pred_temp_df = df[df['date_stamp'].isin(months[test])][['date_stamp', 'symbol', 'fwd_rtn']] \
            .reset_index(drop=True).copy()
        oos_pred_temp_df['y_true'] = y_true
        oos_pred_temp_df['y_hat'] = y_hat
        
        oos_pred_list.append(oos_pred_temp_df)
        
    # Sliding window (end) ----------------------------------------------------------------------------------------------(3)
    
    
    # Aggregate performance metrics & save
    oos_pred_df = pd.concat(oos_pred_list)
    if args.bucket_name == "None":
        oos_pred_df.to_csv(args.destination_file)  # TO DO - toggle write to local vs GCP storage 
    else:
        write_df_gcs_csv(oos_pred_df, "brent_test_bucket", "output.csv")


if __name__ == "__main__":
    # https://gist.github.com/matthewfeickert/3b7d30e408fe4002aac728fc911ced35
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

# cd ~/numpyro_models/numpyro_models
# conda activate pytorch_pyro
# python3 ~/numpyro_models/numpyro_models/m03.py -c m03_config.json