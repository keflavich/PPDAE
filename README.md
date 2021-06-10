# ProtoPlanetary Disk AutoEncoders


### Image Samples


## Usage

Use `ae_main.py` to train a AE model with the following parameters:
```
  -h, --help            show this help message and exit
  --dry-run             Load data and initialize models [False]
  --machine MACHINE     where to is running (local, colab, [exalearn])
  --img-norm IMG_NORM   type of normalization for images ([global], image)
  --par-norm PAR_NORM   physical parameters are 0-1 scaled ([T],F)
  --subset SUBSET       data subset ([25052021],fexp1)
  --part-num PAR_NUM    partition subset number ([1],2,3,4,5)
  --lr LR               learning rate [1e-4]
  --lr-sch LR_SCH       learning rate shceduler ([None], step, exp, cosine,
                        plateau)
  --transfrom TRANSFORM
                        applies transformation to images ([True], False)
  --cond COND           physics conditioned AE ([F],T)
  --feed-phy FEED_PHY   feed physics to decoder (F,[T])
  --dropout DROPOUT     dropout for all layers [0.2]
  --early-stop          Early stoping
  --comment COMMENT     extra comments for runtime labels
  --model-name MODEL_NAME
                        name of model ([Forward_AE], Dev_Forward_AE)
  --batch-size BATCH_SIZE
                        batch size [128]
  --num-epochs NUM_EPOCHS
                        total number of training epochs [100]
```

### Recontruction examples

Training logs and models https://app.wandb.ai/jorgemarpa/PPD-AE/overview

## Sources and inspiration

* https://www.jeremyjordan.me/variational-autoencoders/
* https://github.com/wiseodd/generative-models
