import torch
import yaml
from tqdm.auto import tqdm
from ppdae.ae_model_phy import ConvLinTrans_AE
from ppdae.dataset_large import RobitailleGrid

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

    train_loader, val_loader, test_loader = dataset.get_dataloader(shuffle=False)

    # Iterate through the data using the DataLoader
    for batch in tqdm(train_loader):
        # Move the data to the appropriate device (CPU or GPU)
        inputs = batch[0].to(device)  # Adjust this according to your data structure
        # Perform inference
        with torch.no_grad():
            predictions = model.encode(inputs)

        # You can store the predictions or perform any post-processing here
        all_predictions.append(predictions)

    # Concatenate the predictions if needed
    all_predictions = torch.cat(all_predictions, dim=0)

    globals().update(locals())

if __name__ == "__main__":
    main()
