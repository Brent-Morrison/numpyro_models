import argparse
import torch
import pandas as pd
#import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Get data -----------------------------------------------------------------------------------------------------------------

def get_data(filepath, date_filter):
    
    # CSV
    d = pd.read_csv(filepath, index_col=None)
    d['date_stamp'] = pd.to_datetime(d['date_stamp'], format="%d/%m/%Y")
    
    # Date filter
    min_date = date_filter
    max_date = d['date_stamp'].max()

    idx = d[(d['date_stamp'] > min_date) & (d['date_stamp'] < max_date)]['date_stamp'].values
    Y = d.query('(date_stamp > @min_date) & (date_stamp < @max_date)')['fwd_rtn'].values
    X = d[(d['date_stamp'] > min_date) & (d['date_stamp'] < max_date)].iloc[:, 2:4].values
    X = torch.from_numpy(X).to(torch.float32)
    #X = X.values.to(torch.float32)
    #X = torch.tensor(X.values, dtype=torch.float32).to(torch.device("cpu")) 

    return X, Y, idx



# Define model -------------------------------------------------------------------------------------------------------------

# https://www.tutorialspoint.com/how-to-implementing-an-autoencoder-in-pytorch
# https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#example-ml-model-parameters

class AutoEnc(torch.nn.Module):
   def __init__(self, ncol):
      super().__init__()

      self.encoder = torch.nn.Sequential(
         torch.nn.Linear(ncol, 8),
         torch.nn.ReLU(),
         torch.nn.Linear(8, 4),
         torch.nn.ReLU(),
         torch.nn.Linear(4, 2)
      )

      self.decoder = torch.nn.Sequential(
         torch.nn.Linear(2, 4),
         torch.nn.ReLU(),
         torch.nn.Linear(4, 8),
         torch.nn.ReLU(),
         torch.nn.Linear(8, ncol),
         torch.nn.Sigmoid()
      )
   def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return encoded, decoded


class AutoEncDyn(torch.nn.Module):
    def __init__(self, layer_widths):
        super().__init__()

        # Encoder
        self.encoder = torch.nn.ModuleList()
        encz = zip(layer_widths[:-1], layer_widths[1:])
        *hidden, last = encz
        for n_in, n_out in hidden:
            self.encoder.append(torch.nn.Linear(n_in, n_out))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(last[0], last[1]))

        # Decoder
        self.decoder = torch.nn.ModuleList()
        decz = zip(reversed(layer_widths[1:]), reversed(layer_widths[:-1]))
        *hidden, last = decz
        for n_in, n_out in hidden:
            self.decoder.append(torch.nn.Linear(n_in, n_out))
            self.decoder.append(torch.nn.ReLU())
        self.decoder.append(torch.nn.Linear(last[0], last[1]))
    
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
        


# Instantiate model & training loop ----------------------------------------------------------------------------------------
def model(data, epochs):
    
    # Instantiate model
    #model = AutoEnc(data.shape[1])
    model = AutoEnc1(data.shape[1])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e8)


    # Training loop
    #losses = []
    for epoch in range(epochs):
        encoded, decoded = model(data)
        loss = loss_fn(decoded, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Training error: {loss.item():>7f}")
        #losses.append(loss.detach())

    return encoded



# Main ---------------------------------------------------------------------------------------------------------------------

def main(args):
    
    # Convert date parameter from string to date, for use as data frame filter
    #date_filter = dt.datetime.strptime(args.date_filter, '%Y-%m-%d').date()
    #date_filter = args.date_filter

    # Training data
    X, _, _ = get_data(filepath=args.filepath, date_filter=args.date_filter)

    encoded = model(data=X, epochs=50)

    encoded_np = encoded.detach().numpy()
    pd.DataFrame(encoded_np) \
        .to_csv('/c/Users/brent/Documents/R/Misc_scripts/m02_preds.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression")
    parser.add_argument("-d", "--date_filter", nargs="?", default="2022-12-31", type=str)
    parser.add_argument("-f", "--filepath"   , nargs="?", default="/c/Users/brent/Documents/R/Misc_scripts/stocks.csv", type=str)

    args = parser.parse_args()

    main(args)


# python ~/numpyro_models/numpyro_models/m02.py -d 2021-12-31