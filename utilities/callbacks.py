import os
import time
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from utilities import pickling
from evaluation import compute_average_AUC_from_dictionary
import logging



logger = logging.getLogger(__name__)


class Callback(object):
    """
    Base class for all callbacks
    Defines all available events and default arguments
    All callbacks should define `.callback_name` property
    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def callback_name(self):
        return None

    def on_train_start(self, *args, **kwargs):
        pass
    
    def on_epoch_start(self, epoch, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs = None, neptune_experiment = None, *args, **kwargs):
        pass

    def on_batch_start(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, logs = None, neptune_experiment = None, **kwargs):
        pass

    def on_train_start(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, logs = None, neptune_experiment = None, **kwargs):
        pass

    def on_val_start(self, *args, **kwargs):
        pass

    def on_val_end(self, *args, logs = None, neptune_experiment = None, **kwargs):
        pass
    

def resolve_callbacks(event, callback_list, *args, **kwargs):
    """
    Asks each callback in the callback list to run an event method
    i.e. `on_train_start`. If callback returns an output, it is saved to
    the common dict and returned.

    :param event: Event name to be called, e.g. `on_train_start`
    :param callback_list: List of callback objects
    """

    callback_outputs = {}
    for callback in callback_list:
        event_method = getattr(callback, event)
        result = event_method(*args, **kwargs)
        if result:
            callback_outputs[callback.callback_name] = result
    return callback_outputs


class History(Callback):
    """
    Saves history every train and val loop
    """

    def __init__(self, save_path, save_every_k_examples=-1,
                 distributed=False, local_rank=0, world_size=-1):
        self.save_path = save_path
        self.save_every_k_examples = save_every_k_examples
        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = world_size
        super(History, self).__init__()
    
    @property
    def callback_name(self):
        return "History"

    def on_train_start(self):
        self.history = {}
        self.history_batch = {}
    
    def on_val_start(self):
        self.history = {}
        self.history_batch = {}
    
    def save_history(self, epoch, logs, phase):
        # Saves history every PHASE, i.e. separately for training and validation
        logs = logs or {}

        for k, v in logs.items():
            if k.endswith("labels"):
                # we don't need to save labels every epoch.
                if k not in self.history:
                    self.history[k] = v
            else:
                self.history.setdefault(k, []).append(v)
        
        history_pkl_name = f"history_{self.local_rank}_{phase}.pkl" if self.distributed else f"history_{phase}.pkl"
        batch_pkl_name = f"history_batch_{self.local_rank}_{phase}.pkl" if self.distributed else f"history_batch_{phase}.pkl"

        print(f"\nsave path from rank {self.local_rank}: {self.save_path}\n")
        if self.save_path is not None:
            pickle.dump(self.history, open(os.path.join(self.save_path, history_pkl_name), "wb"))
            if self.save_every_k_examples != -1:
                pickle.dump(self.history_batch, open(os.path.join(self.save_path, batch_pkl_name), "wb"))

    def on_train_end(self, epoch, logs = None, **kwargs):
        self.save_history(epoch, logs, "train")
    
    def on_val_end(self, epoch, logs = None, **kwargs):
        self.save_history(epoch, logs, "val")


class EvaluateEpoch(Callback):
    def __init__(self, save_path, distributed, local_rank,
                 world_size, label_type):
        super(EvaluateEpoch, self).__init__()
        self.save_path = save_path
        self.distributed = distributed
        self.local_rank = local_rank
        self.world_size = world_size
        self.label_type = label_type

    @property
    def callback_name(self):
        return "EvaluateEpoch"
    
    @staticmethod
    def load_pickle_until_no_errors(log_i_path, wait_time):
        # File might exist, but it might be incomplete (EOFError)
        loaded_pkl = None
        while loaded_pkl is None:
            try:
                loaded_pkl = pickling.unpickle_from_file(log_i_path)
            except:
                time.sleep(wait_time)
        return loaded_pkl

    def get_current_log_from_rank_i(self, i, epoch, phase, wait_time=0.1):
        log_i_path = os.path.join(self.save_path, f'history_{i}_{phase}.pkl')
        while not os.path.exists(log_i_path):
            time.sleep(wait_time)
        loaded_pkl = self.load_pickle_until_no_errors(log_i_path, wait_time)
        while loaded_pkl['epoch_number'][-1] != epoch:
            time.sleep(wait_time)
            loaded_pkl = self.load_pickle_until_no_errors(log_i_path, wait_time)
        
        return loaded_pkl
    
    def get_logs_from_world(self, epoch, phase):
        history_pkls = []

        for i in range(self.world_size):
            # Get history log from i-th rank
            current_log = self.get_current_log_from_rank_i(i, epoch, phase)

            # Convert some lists into dicts if list has only one element
            keys_to_convert = [
                #'training_losses',
                'training_predictions',
                'subgroup_df',
                #'validation_losses',
                'validation_predictions',
            ]
            for k in keys_to_convert:
                if k in current_log:
                    if type(current_log[k]) == list:
                        if len(current_log[k]) == 1:
                            current_log[k] = current_log[k][0]
            
            # Add correct history log to the list
            history_pkls.append(current_log)
        
        return history_pkls

    
    def get_predictions_from_world(self, epoch, log_items_names_list, 
                                   logs, phase, get_loss=True):
        history_pkls = self.get_logs_from_world(epoch, phase)
        
        # Convert list of dicts into single dict & more custom stuff
        keys_to_convert = [
            #'training_losses',
            'training_predictions',
            'training_labels',
            #'validation_losses',
            'validation_predictions',
            'validation_labels'
        ]
        history_dict = {}
        for d in history_pkls:
            for ktc in keys_to_convert:
                if ktc not in d:
                    # Key to convert not in history file
                    # Happens when trying to convert training history keys
                    # during validation phase and vice versa
                    continue
                if ktc not in history_dict:
                    history_dict[ktc] = {}
                for k, v in d[ktc].items():
                    history_dict[ktc][k] = v
        
        # Subgroup data frame
        if 'subgroup_df' in history_pkls[0]:
            history_dict['subgroup_df'] = history_pkls[0]['subgroup_df']

        return history_dict

    def evaluate_phase_birads(self, epoch, logs, phase):
        # Multiclass ROC AUC for BI-RADS classification
        # Works also for BPE classification
        if self.distributed and self.world_size > 1:
            # Gather logs from all processes 
            print(f"Get logs from world in phase {phase}")
            predictions_key_names_list = []
            logs_ = self.get_predictions_from_world(
                epoch,
                predictions_key_names_list,
                logs,
                phase,
                get_loss=False
            )
        else:
            logs_ = logs
        
        if phase == 'train':
            labels = np.vstack(list(logs_['training_labels'].values()))
            preds = np.vstack(list(logs_['training_predictions'].values()))
        elif phase == 'val':
            labels = np.vstack(list(logs_['validation_labels'].values()))
            preds = np.vstack(list(logs_['validation_predictions'].values()))
        print("Labels:", labels)
        print("Preds:", preds)

        try:
            auc = roc_auc_score(labels, preds)
        except Exception as err:
            auc = 0.
            logger.warn(f"Caught error when calculating AUROC: {err}")
        print("BI-RADS/BPE AUROC:", auc)

        return auc

    def evaluate_phase(self, epoch, logs, phase):
        if self.distributed and self.world_size > 1:
            # Gather logs from all processes 
            print(f"Get logs from world in phase {phase}")
            predictions_key_names_list = []
            preds = self.get_predictions_from_world(
                epoch,
                predictions_key_names_list,
                logs,
                phase,
                get_loss=False
            )

            # calculate auroc
            if phase == 'train':
                auc_b, auc_m, auc_s, auprc_m = compute_average_AUC_from_dictionary(
                    preds['training_predictions'],
                    preds['training_labels'],
                    preds['subgroup_df']
                )
            elif phase == 'val':
                auc_b, auc_m, auc_s, auprc_m = compute_average_AUC_from_dictionary(
                    preds['validation_predictions'],
                    preds['validation_labels'],
                    preds['subgroup_df']
                )
            else:
                raise ValueError
            
            auc_dict = {
                "auc_benign": auc_b, 
                "auc_malignant": auc_m,
                "auc_subgroup": auc_s,
                "auprc_malignant": auprc_m
            }
            return auc_dict
        else:
            if phase == 'train':
                auc_b, auc_m, auc_s, auprc_m = compute_average_AUC_from_dictionary(
                    logs['training_predictions'],
                    logs['training_labels'],
                    logs['subgroup_df']
                )
            elif phase == 'val':
                auc_b, auc_m, auc_s, auprc_m = compute_average_AUC_from_dictionary(
                    logs['validation_predictions'],
                    logs['validation_labels'],
                    logs['subgroup_df']
                )
            else:
                raise ValueError

            auc_dict = {
                "auc_benign": auc_b, 
                "auc_malignant": auc_m,
                "auc_subgroup": auc_s,
                "auprc_malignant": auprc_m
            }
            return auc_dict

    def evaluate_phase_dcis(self, epoch, logs, phase):
        """ Evaluate accuracy of classifying three separate groups
        """
        preds = np.array(list(logs['training_predictions'].values()))  # (n_samples, 3)  [3 is n_classes]
        labels = np.array(list(logs['training_labels'].values()))  # (n_samples, 3)  [3 is n_classes]

        # Calculate AUROC for each group separately
        try:
            negative_benign_auc = roc_auc_score(labels[:, 0], preds[:, 0])
        except Exception as e:
            print(e)
            negative_benign_auc = 0.0
        try:
            target_group_auc = roc_auc_score(labels[:, 1], preds[:, 1])
        except Exception as e:
            print(e)
            target_group_auc = 0.0
        try:
            invasive_or_dcis3_auc = roc_auc_score(labels[:, 2], preds[:, 2])
        except Exception as e:
            print(e)
            invasive_or_dcis3_auc = 0.0

        return {
            "negative_benign_auc": negative_benign_auc,
            "target_group_auc": target_group_auc,
            "invasive_or_dcis3_auc": invasive_or_dcis3_auc
        }

    def log_to_neptune(self, experiment, auc_dict, phase, epoch):
        experiment.log({f'{phase}/AUROC_benign': auc_dict['auc_benign']})
        experiment.log({f'{phase}/AUROC_malignant': auc_dict['auc_malignant']})
        experiment.log({f'{phase}/AUPRC_malignant': auc_dict['auprc_malignant']})
        if 'auc_subgroup' in auc_dict and type(auc_dict['auc_subgroup']) == dict:
            # Log text for all subgroups
            for k, v in auc_dict['auc_subgroup'].items():
                experiment.log({f'{phase}/AUROC_benign_{k}': v[1]})
                experiment.log({f'{phase}/AUROC_malignant_{k}': v[2]})

    def log_to_neptune_dcis(self, experiment, auc_dict, phase, epoch):
        """Simple logging of results to neptune for DCIS experiment"""

        experiment.log({f'{phase}/AUROC_neg_ben': auc_dict['negative_benign_auc']})
        experiment.log({f'{phase}/AUROC_target': auc_dict['target_group_auc']})
        experiment.log({f'{phase}/AUROC_invasive': auc_dict['invasive_or_dcis3_auc']})

    def on_train_end(self, epoch, logs: dict = None, neptune_experiment = None, **kwargs):
        if self.local_rank != 0 and self.local_rank != -1:  # evaluate only in master process
            return
        
        if self.label_type == 'birads':
            auc = self.evaluate_phase_birads(epoch, logs, "train")
            if neptune_experiment:
                pass  # TODO
            auc_dict = {"auc": auc}
        elif self.label_type == 'bpe':
            auc = self.evaluate_phase_birads(epoch, logs, "train")
            if neptune_experiment:
                pass  # TODO
            auc_dict = {"auc": auc}
        elif self.label_type == 'dcis':
            auc_dict = self.evaluate_phase_dcis(epoch, logs, "train")
            if neptune_experiment:
                self.log_to_neptune_dcis(
                    experiment=neptune_experiment,
                    auc_dict=auc_dict,
                    phase="train",
                    epoch=epoch
                )
        else:
            auc_dict = self.evaluate_phase(epoch, logs, "train")
            if neptune_experiment:
                self.log_to_neptune(
                    experiment=neptune_experiment,
                    auc_dict=auc_dict,
                    phase="train",
                    epoch=epoch
                )
        
        return auc_dict
    
    def on_val_end(self, epoch, logs: dict = None, neptune_experiment = None, **kwargs):
        if self.local_rank != 0 and self.local_rank != -1:  # evaluate only in master process
            return

        if self.label_type == 'birads':
            auc = self.evaluate_phase_birads(epoch, logs, "val")
            if neptune_experiment:
                neptune_experiment.log_metric(f'pretrain_birads_auroc/val', auc)
            auc_dict = {"auc": auc}
        elif self.label_type == 'bpe':
            auc = self.evaluate_phase_birads(epoch, logs, "val")
            if neptune_experiment:
                neptune_experiment.log_metric(f'pretrain_bpe_auroc/val', auc)
            auc_dict = {"auc": auc}
        elif self.label_type == 'dcis':
            auc_dict = self.evaluate_phase_dcis(epoch, logs, "val")
            if neptune_experiment:
                self.log_to_neptune_dcis(
                    experiment=neptune_experiment,
                    auc_dict=auc_dict,
                    phase="val",
                    epoch=epoch
                )
        else:
            auc_dict = self.evaluate_phase(epoch, logs, "val")
            if neptune_experiment:
                self.log_to_neptune(
                    experiment=neptune_experiment,
                    auc_dict=auc_dict,
                    phase="val",
                    epoch=epoch
                )
        
        return auc_dict


class RedundantCallback(Callback):
    """
    This is just a temporary callback that I use to make sure
    that when using >1 callbacks everything works fine.
    """

    def __init__(self, message = None):
        self.training_started = False
        self.current_epoch = 0
        if message:
            print(message)
    
    @property
    def callback_name(self):
        return "RedundantCallback"

    def on_train_start(self, **kwargs):
        self.training_started = True
    
    def on_epoch_start(self, epoch, **kwargs):
        self.current_epoch = epoch

    def ping(self):
        print("PING!")
        return "pong"
