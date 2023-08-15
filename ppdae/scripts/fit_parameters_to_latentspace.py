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

def load_wandb_run(wandbstr, basepath='/blue/adamginsburg/adamginsburg/robitaille/ML_PPDAE'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLinTrans_AE(img_width=300, img_height=400, phy_dim=16,
                            n_conv_blocks=4, latent_dim=16,
                            feed_phy=False).to(device)

    # config seems to be wrong?
    with open(f'{basepath}/wandb/{wandbstr}/files/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    #for key, value in config.items():
    #    setattr(model, key, value)

    modelfile = f'{basepath}/wandb/{wandbstr}/files/model.pt'
    if device.type == 'cpu':
        model_parameters = torch.load(modelfile, map_location=torch.device('cpu'))
    else:
        model_parameters = torch.load(modelfile)
    model.load_state_dict(model_parameters)

    return model

def rf_regressor(all_predictions, all_params):
    all_predictions_cpu = all_predictions.cpu()
    all_params_cpu = all_params.cpu()
    XX = all_predictions_cpu
    YY = all_params_cpu
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

    return rf_regressor


def load_predictions(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize lists to store your predictions
    all_predictions = []
    all_params = []

    #train_loader, val_loader, test_loader = dataset.get_dataloader(shuffle=False)
    indices = np.arange(dataset.imgs_memmaps[0].shape[0])
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=100, sampler=sampler, drop_last=False)
    testdataloader = DataLoader(TensorDataset(
        torch.Tensor(dataset.imgs_test),
        torch.Tensor(dataset.par_test)))

    model.eval()
    # Iterate through the data using the DataLoader
    for batch in tqdm(dataloader):
        # Move the data to the appropriate device (CPU or GPU)
        inputs = batch[0].to(device)# Adjust this according to your data structure
        phy = batch[1].to(device)
        # Perform inference
        with torch.no_grad():
            predictions = model.encode(inputs, phy=phy)

        all_predictions.append(predictions)
        all_params.append(phy)

    # Concatenate the predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_params = torch.cat(all_params, dim=0)

    return all_predictions, all_params

def load_everything(wandb_str='spubsmi_10000'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    geometry = 'spubsmi'
    dataset = RobitailleGrid(machine='hpg', transform=False, par_norm=False,
                             subset=geometry, image_norm='image',)

    model = load_wandb_run(wandb_str)

    all_predictions, all_params = load_predictions(model, dataset)

    regressor = rf_regressor(all_predictions, all_params)

    return model, dataset, all_predictions, regressor

def predict_from_parameters(parameters, regressor, model=load_wandb_run('spubsmi_10000')):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_pars = regressor.predict(parameters)

    #zz = torch.from_numpy(np.concatenate([latent_pars.squeeze(), parameters.squeeze()])).float().to(device)
    zz = torch.from_numpy(latent_pars).float().to(device)
    parameters = torch.from_numpy(parameters).float().to(device)

    model.eval()

    with torch.no_grad():
        img = model.decode(zz, phy=parameters)

    return img.squeeze()

def main():
    model, dataset, all_predictions, regressor = load_everything()

    #pars_arr = np.array(np.array(pars[15000]['star.radius', 'star.temperature', 'disk.mass', 'disk.rmax', 'disk.beta', 'disk.p', 'disk.h100', 'envelope.rho_0', 'envelope.rc', 'cavity.power', 'cavity.theta_0', 'cavity.rho_0', 'ambient.density', 'ambient.temperature', 'scattering', 'inclination']).tolist())
    #pars_arr = torch.tensor(pars_arr, dtype=torch.float32)
    example = predict_from_parameters(np.array([[ 4.1610e-01,  9.7610e+03,  6.7770e-03,
                                       2.8940e+02,  1.0070e+00, -1.2050e+00,
                                       4.2020e+00,  3.7470e-17,  2.8940e+02,
                                       1.0620e+00, 2.2050e+01,  6.7120e-22,
                                       1.0000e-23,  1.0000e+01,  1.0000e+00,
                                       6.7665e+01]]),
                                      regressor)
    globals().update(locals())

if __name__ == "__main__":
    main()
