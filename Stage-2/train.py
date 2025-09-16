import pytorch_lightning as pl
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """
    @staticmethod
    def gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("train/grad_norm", self.gradient_norm(model))

def train(config):
    pl.seed_everything(config.seed_everything, workers=True)

    if config.trainer.resume_from_checkpoint is not None:
        print("Resuming from checkpoint: ", config.trainer.resume_from_checkpoint)
        model_module = LitCoMER.load_from_checkpoint(config.trainer.resume_from_checkpoint)
    else:
        print("Training from new weights")
        model_module = LitCoMER(
            d_model = config.model.d_model,
            # encoder
            growth_rate = config.model.growth_rate,
            num_layers = config.model.num_layers,
            # decoder
            nhead = config.model.nhead,
            num_decoder_layers = config.model.num_decoder_layers,
            dim_feedforward = config.model.dim_feedforward,
            dropout = config.model.dropout,
            dc = config.model.dc,
            cross_coverage = config.model.cross_coverage,
            self_coverage = config.model.self_coverage,
            # beam search
            beam_size = config.model.beam_size,
            max_len = config.model.max_len,
            alpha = config.model.alpha,
            early_stopping = config.model.early_stopping,
            temperature = config.model.temperature,
            # training
            learning_rate = config.model.learning_rate,
            patience = config.model.patience,
        )
    
    if config.model.path is not None and config.trainer.resume_from_checkpoint is None:
        model_module.load_from_pretrained(config.model.path)

    logger = Logger(config.logging.experiment, 
                    project=config.logging.project, 
                    config=dict(config), log_model='all')
    logger.watch(model_module.comer_model, log="all", log_freq=100)

    lr_callback = LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback_all = ModelCheckpoint(dirpath="checkpoints/all_epochs",
                                                filename="{epoch}-{step}-{val_ExpRate:.4f}", 
                                                save_top_k=-1,
                                                every_n_epochs=1,)

    checkpoint_callback_best = ModelCheckpoint(save_top_k=config.trainer.callbacks[1].init_args.save_top_k, 
                                                    monitor=config.trainer.callbacks[1].init_args.monitor, 
                                                    mode=config.trainer.callbacks[1].init_args.mode,
                                                    filename=config.trainer.callbacks[1].init_args.filename)
    data_module = CROHMEDatamodule(
        zipfile_path = config.data.zipfile_path,
        test_year = config.data.test_year,
        train_batch_size = config.data.train_batch_size,
        eval_batch_size = config.data.eval_batch_size,
        num_workers = config.data.num_workers,
        scale_aug = config.data.scale_aug,)
    
    grad_norm_callback = GradNormCallback()
    
    trainer = pl.Trainer(
        devices=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        val_check_interval=1.0,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        deterministic=config.trainer.deterministic,
        callbacks = [lr_callback, 
                     grad_norm_callback,
                     checkpoint_callback_best,
                     checkpoint_callback_all],
        resume_from_checkpoint = config.trainer.resume_from_checkpoint,
    )

    trainer.fit(model_module, 
                data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    print(config.dumps())
    train(config)