import numpy as np
from tqdm.auto import tqdm
import os

from astropy import log
from astropy import units as u
from astropy.table import Table
import hyperion.model

from sklearn.model_selection import train_test_split


def make_test_data(geometry, pars, imgs, rootdir='/orange/adamginsburg/robitaille_models/ML_PPDAE/'):
    print(f"Beginning setting up training data for {geometry}")
    train_idx, test_idx = train_test_split(np.arange(len(pars)),
                                           test_size=.2, random_state=99)


    # parameters_to_fit = ['star.radius', 'star.temperature', 'envelope.rho_0',
    #  'envelope.power', 'ambient.density', 'ambient.temperature', 'scattering',
    # ]

    # include everything _up to_ model luminosity
    last = pars.colnames.index('Model Luminosity')
    first = 1 # skip MODEL_NAME
    parameters_to_fit = pars.colnames[first:last]

    if imgs.shape[2] == 1:
        # if the images are 2d
        parameters_to_fit.remove('inclination')

    print(f"Fit parameters are {parameters_to_fit}")
    print(f"There are {len(parameters_to_fit)} parameters")

    pars_arrlike = np.array(pars[parameters_to_fit]).view(float).reshape(len(pars), len(parameters_to_fit))
    np.save(f'{rootdir}/param_arr_gridandfiller_{geometry}_train_all.npy',
            pars_arrlike[train_idx].astype("float32"),
           )
    np.save(f'{rootdir}/param_arr_gridandfiller_{geometry}_test.npy',
            pars_arrlike[test_idx].astype("float32")
           )
    np.save(f'{rootdir}/img_array_gridandfiller_imagenorm_{geometry}_train_all.npy', imgs[train_idx].astype("float32"))
    np.save(f'{rootdir}/img_array_gridandfiller_imagenorm_{geometry}_test.npy', imgs[test_idx].astype("float32"))
    np.save(f'{rootdir}/{geometry}_parnames.npy',
            parameters_to_fit)
    print(f"Saved '{rootdir}/img_array_gridandfiller_imagenorm_{geometry}_test.npy'")
    print(f"Done setting up training data for {geometry} with training shape {train_idx.shape} and testing shape {test_idx.shape}")


def setup_training_for_geometry(
    geometry='spubsmi',
    basepath='/blue/adamginsburg/richardson.t/research/flux/',
    parsversion='1.2',
    gridversion='1.1',
    max_rows=None,
    rootdir='/orange/adamginsburg/robitaille_models/ML_PPDAE/'
):
    """
    hyperion models can only be read from relative paths
    """

    os.chdir(basepath)
    gridpath = f'{basepath}/grids-{gridversion}/{geometry}/output'

    pars = Table.read(f'{basepath}/robitaille_models-{parsversion}/{geometry}/augmented_parameters.fits')
    npars = len(pars)

    # read a single model to get metadata
    mn = pars['MODEL_NAME'][0]
    tem = hyperion.model.ModelOutput(f'{gridpath}/{mn[:2]}/{mn[:-3]}.rtout').get_quantities()['temperature'][0].array
    temshape = tem.shape

    if max_rows is not None and npars > max_rows:
        nrows = max_rows
        training_filename = f'{rootdir}/{geometry}_temperature_grid_for_ML_rows{max_rows}.npy'
    else:
        nrows = npars
        training_filename = f'{rootdir}/{geometry}_temperature_grid_for_ML.npy'

    filesize = (nrows * np.product(temshape) * 4 * u.byte).to(u.GB)
    print(f"training file will have size {filesize}")

    if not os.path.exists(training_filename):
        # this takes about 20 minutes for 10k parameters
        from astropy import log
        log.setLevel(0)

        arr = np.memmap(training_filename,
                        shape=(nrows, *tem.shape),
                        dtype=np.float32,
                        mode='w+')

        for ii, mn in tqdm(enumerate(pars['MODEL_NAME'][:nrows])):
            arr[ii, :, :, :] = (model
                                .ModelOutput(f'{gridpath}/{mn[:2].lower()}/{mn[:-3]}.rtout')
                                .get_quantities()['temperature'][0].array
                               )

        log.setLevel('INFO')
    else:
        arr = np.memmap(training_filename,
                        shape=(nrows, *tem.shape),
                        dtype=np.float32,
                        mode='r+')

    make_test_data(geometry, pars, arr, rootdir=rootdir)

def main(rootdir='/orange/adamginsburg/robitaille_models/ML_PPDAE/'):

    import runpy
    scriptpath = "{rootdir}/PPDAE/ppdae/scripts/ae_main.py"

    for max_rows in (10000, None):
        for geometry in ('spubsmi', 'spubhmi'):
            print(f"Setting up {geometry} with limit {max_rows}")
            setup_training_for_geometry(rootdir=rootdir, max_rows=max_rows, geometry=geometry)
            print(f"Running training for {geometry} with limit {max_rows}")
            runpy.run_module(mod_name='main',
                             run_name=scriptpath,
                             argv="--latent-dim 16 --batch-size 128 --machine hpg --data Robitaille --subset='spubsmi'".split())
            print(f"done running training for {geometry} with limit {max_rows}")
    #%run $rootdir/PPDAE/ppdae/scripts/ae_main.py --latent-dim 16 --batch-size 128 --machine hpg --data Robitaille --subset='spubsmi'

if __name__ == "__main__":
    main()
    #sbatch --job-name=gpu-ppdae --account=astronomy-dept --qos=astronomy-dept -p gpu --gpus=a100:1 --ntasks=1 --cpus-per-task=8 --nodes=1 --mem=64gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /orange/adamginsburg/robitaille_models/ML_PPDAE/PPDAE/ppdae/scripts/setup_training.py"
