# ProtoPlanetary Disk AutoEncoders


### Image Samples


## Usage

Use `ae_main.py` to train a AE model with the following parameters:
```
  -h, --help            show this help message and exit
  --dry-run             Load data and initialize models [False]
  --machine MACHINE     were to is running (local, colab, [exalearn])
  --lr LR               learning rate [1e-4]
  --lr-sch LR_SCH       learning rate shceduler ([None], step,
                        exp,cosine, plateau)
  --batch-size BATCH_SIZE
                        batch size [128]
  --num-epochs NUM_EPOCHS
                        total number of training epochs [100]
  --early-stop          Early stoping
  
  --dropout DROPOUT     dropout for lstm/tcn layers [0.2]
  --part-num PAR_NUM    partition batch number ([1],2,3,4,5)
  --comment COMMENT     extra comments
```

### Recontruction examples

Training logs and models https://app.wandb.ai/jorgemarpa/PPD-AE/overview

## Sources and inspiration

* https://www.jeremyjordan.me/variational-autoencoders/
* https://github.com/wiseodd/generative-models
