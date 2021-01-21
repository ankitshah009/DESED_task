import warnings
import scipy

import torch.nn as nn
import torch
import pytorch_lightning as pl

from utils_model.RNN import BidirectionalGRU
from utils_model.CNN import CNN
from utils.utils import AverageMeterSet

from utils import ramps
import numpy as np
import pandas as pd
from training import update_ema_variables
from dcase_util.data import ProbabilityEncoder
from utils.ManyHotEncoder import ManyHotEncoder


class MeanTaecher(pl.LightningModule):

    def __init__(self, 
        adjust_lr=True,
        rampdown_value=1, 
        rampup_value=None, 
        max_learning_rate=0.001, 
        sample_rate = 16000, 
        pooling_time_ratio = 4,
        hop_size = 255,
        median_window = 0.45, 
        thresholds = [0.5],
        model = None,
        ema_model = None,
        train_loader_length=None,
        dataloader=None,
        max_consistency_cost=2,
        decoder=None,
        mask_strong = None,
        mask_weak = None,
        meters = None,
        **optim_kwargs,
        ):

        #data 

        self.sample_rate = sample_rate
        self.pooling_time_ratio = pooling_time_ratio
        self.hop_size = hop_size
        self.median_window = median_window
        self.thresholds = thresholds

        self.adjust_lr = adjust_lr
        self.rampdown_value = rampdown_value
        self.rampup_value = rampup_value
        self.max_learning_rate = max_learning_rate
        self.max_consistency_cost = max_consistency_cost
        self.optim_kwargs = optim_kwargs
        self.decoder = decoder

        #models 
        self.model = model
        self.ema_model = ema_model
        self.train_loader_length =train_loader_length
        self.mask_strong = mask_strong
        self.mask_weak = mask_weak

        # meters 
        self.meters = AverageMeterSet()

        self.dataloader = dataloader
        self.prediction_dfs = {}
        for threshold in thresholds:
            self.prediction_dfs[threshold] = pd.DataFrame()
        
    
    def forward(self, x):
        
        batch_input, ema_batch_input = x
        
        # Getting prediction from teacher model
        strong_pred_ema, weak_pred_ema = self.ema_model.forward(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        # Getting predictions from student model
        strong_pred, weak_pred = self.model.forward(batch_input)

        return strong_pred_ema, weak_pred_ema, strong_pred, weak_pred

    
    def calculate_loss(self, 
        strong_pred_ema, 
        weak_pred_ema, 
        strong_pred, 
        weak_pred, 
        target
        ):

        class_criterion = nn.BCELoss()
        consistency_criterion = nn.MSELoss()

        # TODO: Move to cuda?

        target_weak = target.max(-2)[0]  # Take the max in the time axis

        if self.mask_weak is not None:
            weak_class_loss = class_criterion(
                weak_pred[self.mask_weak], target_weak[self.mask_weak]
            )
            
            self.meters.update("weak_class_loss", weak_class_loss.item())

            ema_class_loss = class_criterion(
                weak_pred_ema[self.mask_weak], target_weak[self.mask_weak]
            )

            self.meters.update("Weak EMA loss", ema_class_loss.item())

            loss = weak_class_loss

            #if i == 0:
            #   log.debug(
            #   f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
            #   f"Target weak mask: {target_weak[mask_weak]} \n "
            #   f"Target strong mask: {target[mask_strong].sum(-2)}\n"
            #   f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
            #   f"tensor mean: {batch_input.mean()}"
            #)

        # Strong BCE loss
        if self.mask_strong is not None:

            strong_class_loss = class_criterion(
                strong_pred[self.mask_strong], target[self.mask_strong]
            )

            self.meters.update("Strong loss", strong_class_loss.item())

            strong_ema_class_loss = class_criterion(
                strong_pred_ema[self.mask_strong], target[self.mask_strong]
            )
            self.meters.update("Strong EMA loss", strong_ema_class_loss.item())

            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if self.ema_model is not None:

            consistency_cost = self.max_consistency_cost * self.rampup_value
            self.meters.update("Consistency weight", consistency_cost)

            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(
                strong_pred, strong_pred_ema
            )

            self.meters.update("Consistency strong", consistency_loss_strong.item())

            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(
                weak_pred, weak_pred_ema
            )
            self.meters.update("Consistency weak", consistency_loss_weak.item())

            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (
            np.isnan(loss.item()) or loss.item() > 1e5
        ), "Loss explosion: {}".format(loss.item())
        assert not loss.item() < 0, "Loss problem, cannot be negative"

        # update loss value
        self.meters.update("Loss", loss.item())

        return loss

    def training_step(self, train_batch, batch_idx):

        loss = None
        
        # TODO: Global step is a property maybe?
        #global_step = self.current_epoch * len(train_loader) + batch_idx
        global_step = self.current_epoch * self.train_loader_length + batch_idx

        #self.rampup_value = ramps.exp_rampup(global_step, self.n_epoch_rampup * len(train_loader))
        self.rampup_value = ramps.exp_rampup(global_step, self.n_epoch_rampup * self.train_loader_length)

        #if self.adjust_lr:
         #   adjust_learning_rate(optimizer, self.rampup_value, self.max_learning_rate)

        self.update("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        #batch_input, ema_batch_input, target = to_cuda_if_available(
         #   batch_input, ema_batch_input, target
        #)

        #((batch_input, ema_batch_input), target) = train_batch
        batch_input, target = train_batch
        strong_pred_ema, weak_pred_ema, strong_pred, weak_pred = self.forward(batch_input)

        # Getting predictions from teacher model
        #strong_pred_ema, weak_pred_ema = self.ema_model.forward(ema_batch_input)
        #strong_pred_ema = strong_pred_ema.detach()
        #weak_pred_ema = weak_pred_ema.detach()

        # Getting predictions from student model
        #strong_pred, weak_pred = self.model.forward(batch_input)

        loss = self.calculate_loss(strong_pred_ema, 
            weak_pred_ema, 
            strong_pred, 
            weak_pred, 
            target)

        self.global_step += 1

        if self.ema_model is not None:
            update_ema_variables(self.model, self.ema_model, 0.999, self.global_step)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        
        list_predictions = []
        for key in self.prediction_dfs:
            list_predictions.append(self.prediction_dfs[key])

        if len(list_predictions) == 1:
            list_predictions = list_predictions[0]

        self.list_predictions = list_predictions


    def validation_step(self, val_batch, batch_idx):

        ((input_data, _), indexes) = val_batch
        
        indexes = indexes.numpy()
        #input_data = to_cuda_if_available(input_data)

        #with torch.no_grad():
        pred_strong, _ = self.model(input_data)
        
        pred_strong = pred_strong.cpu()
        pred_strong = pred_strong.detach().numpy()

        #if i == 0:
        #    logger.debug(pred_strong)

        # Post processing and put predictions in a dataframe
        for j, pred_strong_it in enumerate(pred_strong):
            
            for threshold in self.thresholds:
                pred_strong_bin = ProbabilityEncoder().binarization(
                    pred_strong_it,
                    binarization_type="global_threshold",
                    threshold=threshold,
                )
                pred_strong_m = scipy.ndimage.filters.median_filter(
                    pred_strong_bin, (self.median_window, 1)
                )
                pred = self.decoder(pred_strong_m)
                pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
                # Convert them in seconds
                pred.loc[:, ["onset", "offset"]] *= self.pooling_time_ratio / (
                    self.sample_rate / self.hop_size
                )
                pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(
                    0, self.max_len_seconds
                )

                pred["filename"] = self.dataloader.dataset.filenames.iloc[indexes[j]]
                self.prediction_dfs[threshold] = self.prediction_dfs[threshold].append(
                    pred, ignore_index=True
                )

                #if i == 0 and j == 0:
                 #   logger.debug("predictions: \n{}".format(pred))
                  #  logger.debug("predictions strong: \n{}".format(pred_strong_it))

    
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
        # TODO: Ask Nicolas why we are on the other way around the no_grad and loss.backward (is the order correct)
        
        if self.adjust_lr:
            lr = self.rampup_value * self.rampdown_value * self.max_learning_rate
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), self.optim_kwargs)
        return optimizer
