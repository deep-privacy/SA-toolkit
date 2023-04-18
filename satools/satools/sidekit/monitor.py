"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher, Pierre Champion
"""

import logging
import numpy


class TrainingMonitor:
    """
    Class of object that are used for x-vector model training to monitor accuracy, eer and loss
    """

    def __init__(
        self,
        log_interval=10,
        patience=numpy.inf,
        best_accuracy=0.0,
        best_eer_epoch=1,
        best_eer=100,
        compute_test_eer=False,
    ):
        """
        Constructor method

        :param log_interval: number of batches between two consecutive logs
        :param patience: patience of the training algorithm
        :param best_accuracy: value of the best accuracy obtained during training
        :param best_eer_epoch: Number of the epoch at which the best EER has been obtained
        :param best_eer: value of the best eer obtained during training
        :param compute_test_eer: boolean, if True, compute the EER on the test set
        """
        self.current_epoch = 0
        self.log_interval = log_interval
        self.init_patience = patience
        self.current_patience = patience
        self.best_accuracy = best_accuracy
        self.best_eer_epoch = best_eer_epoch
        self.best_eer = best_eer
        self.compute_test_eer = compute_test_eer
        self.test_eer = []
        self.test_metric = []

        self.training_loss = []
        self.training_acc = []

        self.val_loss = []
        self.val_acc = []
        self.val_eer = []

        self.is_best = True
        self.sw = None

        self.lr_step = 0

    def add_tensorboard(self, path):
        from torch.utils.tensorboard import SummaryWriter
        self.sw = SummaryWriter(path)
        layout = {
            "Test metrics": {
                "bootci test EER": [
                    "Margin",
                    ["test/eer", "test/eer_lower", "test/eer_upper"],
                ],
                "bootci test EER as-norm": [
                    "Margin",
                    ["test/eer_asnorm", "test/eer_lower_asnorm", "test/eer_upper_asnorm"],
                ],
                "test min_cllr": [
                    "Multiline",
                    ["test/min_cllr_m", "test/min_cllr_asnorm_m"],
                ],
                "test EER": [
                    "Multiline",
                    ["test/eer_m", "test/eer_asnorm_m"],
                ],
            },
            "Train metrics": {
                "train accuracy": [
                    "Multiline",
                    ["train/accuracy"],
                ],
                "train loss": [
                    "Multiline",
                    ["train/loss"],
                ],
                "train lr": [
                    "Multiline",
                    ["train/lr"],
                ],
            },
            "Validation metrics": {
                "validation accuracy": [
                    "Multiline",
                    ["validation/accuracy"],
                ],
                "validation loss": [
                    "Multiline",
                    ["validation/loss"],
                ],
                "validation EER": [
                    "Multiline",
                    ["validation/eer"],
                ],
            }
        }
        self.sw.add_custom_scalars(layout)
        return self.sw


    def display(self, add_to_tensorboard=True):
        """
        Display validation and test indicators during the training
        """

        if add_to_tensorboard and self.sw:
            self.sw.add_scalar("validation/accuracy", self.val_acc[-1], len(self.val_acc))
            self.sw.add_scalar("validation/eer", self.val_eer[-1], len(self.val_eer))
            self.sw.add_scalar("validation/loss", self.val_loss[-1], len(self.val_loss))

        logging.info(
            f"`***Validation metrics - Accuracy: {round(self.val_acc[-1], 3)}, EER: {round(self.val_eer[-1], 3)}, Loss: {round(self.val_loss[-1], 3)}***`  "
        )
        if self.compute_test_eer:
            metrics = self.test_metric[-1]
            eer_mean = round(metrics['eer'], 3)
            if metrics['eer_upper'] < 10:
                eer_std = round((metrics['eer_upper'] - metrics['eer_lower'])/2, 3)
            else:
                eer_std = round((metrics['eer'] - metrics['eer_lower']), 3)
            min_cllr = round(metrics['min_cllr'], 3)

            if add_to_tensorboard and self.sw:
                self.sw.add_scalar("test/eer", metrics['eer'], len(self.test_eer))
                self.sw.add_scalar("test/eer_m", metrics['eer'], len(self.test_eer))
                if metrics['eer_upper'] < 10:
                    self.sw.add_scalar("test/eer_upper", metrics['eer_upper'], len(self.test_eer))
                else:
                    self.sw.add_scalar("test/eer_upper", metrics['eer']+eer_std, len(self.test_eer))
                self.sw.add_scalar("test/eer_lower", metrics['eer_lower'], len(self.test_eer))
                self.sw.add_scalar("test/min_cllr", metrics['min_cllr'], len(self.test_eer))
                self.sw.add_scalar("test/min_cllr_m", metrics['min_cllr'], len(self.test_eer))

            if 'asnorm' in metrics:
                asnorm = metrics['asnorm']
                asnorm_eer_mean = round(asnorm['eer'], 3)
                if asnorm['eer_upper'] < 10:
                    asnorm_eer_std = round((asnorm['eer_upper'] - asnorm['eer_lower'])/2, 3)
                else:
                    asnorm_eer_std = round((asnorm['eer'] - asnorm['eer_lower']), 3)
                asnorm_min_cllr = round(asnorm['min_cllr'], 3)

                if add_to_tensorboard and self.sw:
                    self.sw.add_scalar("test/eer_asnorm", metrics["asnorm"]['eer'], len(self.test_eer))
                    self.sw.add_scalar("test/eer_asnorm_m", metrics["asnorm"]['eer'], len(self.test_eer))
                    if asnorm['eer_upper'] < 10:
                        self.sw.add_scalar("test/eer_upper_asnorm", metrics["asnorm"]['eer_upper'], len(self.test_eer))
                    else:
                        self.sw.add_scalar("test/eer_upper_asnorm", asnorm['eer']+asnorm_eer_std, len(self.test_eer))
                    self.sw.add_scalar("test/eer_lower_asnorm", metrics["asnorm"]['eer_lower'], len(self.test_eer))
                    self.sw.add_scalar("test/min_cllr_asnorm", metrics["asnorm"]['min_cllr'], len(self.test_eer))
                    self.sw.add_scalar("test/min_cllr_asnorm_m", metrics["asnorm"]['min_cllr'], len(self.test_eer))

                logging.info(f"`***Test metrics - EER: {eer_mean} ± {eer_std}, min_cllr: {min_cllr} // as-norm - EER: {asnorm_eer_mean} ± {asnorm_eer_std}, min_cllr: {asnorm_min_cllr}***`  ")

            else:
                logging.info(f"`***Test metrics - EER: {eer_mean} ± {eer_std}, min_cllr: {min_cllr}***`  ")

    def display_final(self):
        """
        Display validation and test indicators at the end of thre training process
        """
        logging.info(
            f"Best model {round(self.best_eer, 3)} obtained at epoch {self.best_eer_epoch+1}"
        )

    def update(
        self,
        epoch=None,
        training_acc=None,
        training_loss=None,
        test_eer=None,
        test_metric=None,
        val_eer=None,
        val_loss=None,
        val_acc=None,
        lr=None,
    ):
        """
        Update the content of the Monitor

        :param epoch: number of the current epoch
        :param training_acc:  training accuracy obtained at this epoch
        :param training_loss:  training loss  obtained at this epoch
        :param test_eer: eer obtained on the test set
        :param val_eer: eer obtained on the validation set
        :param val_loss:  loss obtained for the validation set
        :param val_acc:  accuracy obtained on the validation set
        """
        if epoch is not None:
            self.current_epoch = epoch
        if training_acc is not None:
            if self.sw: self.sw.add_scalar("train/accuracy", training_acc, len(self.training_acc))
            self.training_acc.append(training_acc)
        if training_loss is not None:
            if self.sw: self.sw.add_scalar("train/loss", training_loss, len(self.training_loss))
            self.training_loss.append(training_loss)
        if val_eer is not None:
            self.val_eer.append(val_eer)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_acc.append(val_acc)

        if test_metric is not None:
            self.test_metric.append(test_metric)

        if lr is not None:
            if self.sw: self.sw.add_scalar("train/lr", lr, self.lr_step)
            self.lr_step += 1

        # remember best accuracy and save checkpoint
        if self.compute_test_eer and test_eer is not None:
            self.test_eer.append(test_eer)
            self.is_best = test_eer < self.best_eer
            self.best_eer = min(test_eer, self.best_eer)
            if self.is_best:
                self.best_eer_epoch = self.current_epoch
                self.current_patience = self.init_patience
            else:
                self.current_patience -= 1
        elif val_eer is not None:
            self.is_best = val_eer < self.best_eer
            self.best_eer = min(val_eer, self.best_eer)
            if self.is_best:
                self.best_eer_epoch = self.current_epoch
                self.current_patience = self.init_patience
            else:
                self.current_patience -= 1

    def __getstate__(self):
        # Return a dictionary that contains all the attributes of the object,
        # except for the SummaryWriter object which cannot be serialized
        state = self.__dict__.copy()
        del state["sw"]
        return state

    def __setstate__(self, state):
        # Update the current object's attributes with the contents of the serialized dictionary
        self.__dict__.update(state)

