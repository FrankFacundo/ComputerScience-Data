from argparse import ArgumentParser

import pytorch_lightning as pl
from data.data_download import Config
from data_formatters.electricity import ElectricityFormatter
from tft import TemporalFusionTransformer

config = Config("data", "data/electricity.csv")

# download_electricity(config)


data_formatter = ElectricityFormatter()

params = data_formatter.get_experiment_params()
params.update(data_formatter.get_default_model_params())

parser = ArgumentParser(add_help=False)


for k in params:
    if type(params[k]) in [int, float]:
        parser.add_argument("--{}".format(k), type=type(params[k]), default=params[k])
    else:
        parser.add_argument("--{}".format(k), type=str, default=str(params[k]))
hparams_namespace = parser.parse_known_args()[0]
hparams_dict = vars(hparams_namespace)

tft = TemporalFusionTransformer(**hparams_dict)  # .to(DEVICE)

trainer = pl.Trainer(
    max_epochs=tft.num_epochs,
    gradient_clip_algorithm="norm",
    gradient_clip_val=tft.max_gradient_norm,
    overfit_batches=0.01,
)
trainer.fit(tft)

trainer.test()
