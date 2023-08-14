import torch
import numpy as np
import yaml
from tqdm.auto import tqdm
from ppdae.ae_model_phy import ConvLinTrans_AE
from ppdae.dataset_large import RobitailleGrid
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_wandb_run(wandbstr, model, basepath='/blue/adamginsburg/adamginsburg/robitaille/ML_PPDAE'):
    with open(f'{basepath}/wandb/{wandbstr}/files/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    for key, value in config.items():
        setattr(model, key, value)

    modelfile = f'{basepath}/wandb/{wandbstr}/files/model.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        model_parameters = torch.load(modelfile, map_location=torch.device('cpu'))
    else:
        model_parameters = torch.load(modelfile)
    model.load_state_dict(model_parameters)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    geometry = 'spubsmi'
    model = ConvLinTrans_AE(img_width=300, img_height=400).to(device)
    dataset = RobitailleGrid(machine='hpg', transform=False, par_norm=False,
                             subset=geometry, image_norm='image',)


    # Initialize lists to store your predictions
    all_predictions = []

    #train_loader, val_loader, test_loader = dataset.get_dataloader(shuffle=False)
    indices = np.arange(dataset.imgs_memmaps[0].shape[0])
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=100, sampler=sampler, drop_last=False)
    testdataloader = DataLoader(TensorDataset(torch.Tensor(dataset.imgs_test),
                                torch.Tensor(dataset.par_test)))

    # Iterate through the data using the DataLoader
    for batch in tqdm(dataloader):
        # Move the data to the appropriate device (CPU or GPU)
        inputs = batch[0].to(device)  # Adjust this according to your data structure
        # Perform inference
        with torch.no_grad():
            predictions = model.encode(inputs)

    # Concatenate the predictions
    all_predictions = torch.cat(all_predictions, dim=0)


    all_predictions_cpu = all_predictions.cpu()
    XX = all_predictions_cpu[:,:16]
    YY = all_predictions_cpu[:,16:]
    # Assuming you have your data loaded into XX (features) and YY (target)
    XX_train, XX_test, YY_train, YY_test = train_test_split(XX, YY, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor
    # Telkamp used 100 trees, so we will
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    rf_regressor.fit(XX_train, YY_train)

    # Predict on the test data
    YY_pred = rf_regressor.predict(XX_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(YY_test, YY_pred)
    print(f"Mean Squared Error: {mse}")

    globals().update(locals())

if __name__ == "__main__":
    main()
