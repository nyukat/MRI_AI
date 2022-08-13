import numpy as np
import pandas as pd
import math
import sklearn.metrics
import logging



logger = logging.getLogger(__name__)


def compute_average_AUC_with_keys(predictions, labels, keys, report_n=False, auprc=False):
    """
    Compute AUC ROC
    Assumes either breast- or image-level based on the number of predictions
    provided. If there are 4 preds/labels: breast-level prediction.
    If there are 2 preds/labels: image-level prediction.
    This expects predictions and labels to be given in the following order:
    (left_benign, left_malignant, right_benign, right_malignant)
    and for the image-level (benign, malignant).
    """
    assert isinstance(keys, list)
    assert isinstance(predictions, dict)
    
    try:
        if len(predictions[keys[0]]) == 4:
            level = 'breast'
        elif len(predictions[keys[0]]) == 2:
            level = 'image'
        else:
            raise ValueError(f"Please provide either 2 or 4 predictions for evaluation")
    except Exception as why:
        print("Evaluation failed at the very beginning, because: ", why)
        print("Keys:", keys)
        print(type(keys))
        print("Predictions:", predictions)
        print(type(predictions))


    benign_score = []
    benign_label = []
    malignant_score = []
    malignant_label = []

    for i in keys:
        if level == 'breast':
            tmp = max(predictions[i][0], 0)
            score = predictions[i][0] - \
                (tmp + np.logaddexp(-tmp, predictions[i][0] - tmp))
            benign_score.append(score)

            tmp = max(predictions[i][2], 0)
            score = predictions[i][2] - \
                (tmp + np.logaddexp(-tmp, predictions[i][2] - tmp))
            benign_score.append(score)

            tmp = max(predictions[i][1], 0)
            score = predictions[i][1] - \
                (tmp + np.logaddexp(-tmp, predictions[i][1] - tmp))
            malignant_score.append(score)

            tmp = max(predictions[i][3], 0)
            score = predictions[i][3] - \
                (tmp + np.logaddexp(-tmp, predictions[i][3] - tmp))
            malignant_score.append(score)

            benign_label.append(labels[i][0])
            benign_label.append(labels[i][2])
            malignant_label.append(labels[i][1])
            malignant_label.append(labels[i][3])
        
        elif level == 'image':
            tmp = max(predictions[i][0], 0)
            score = predictions[i][0] - \
                (tmp + np.logaddexp(-tmp, predictions[i][0] - tmp))
            benign_score.append(score)

            tmp = max(predictions[i][1], 0)
            score = predictions[i][0] - \
                (tmp + np.logaddexp(-tmp, predictions[i][1] - tmp))
            malignant_score.append(score)

            benign_label.append(labels[i][0])
            malignant_label.append(labels[i][1])

    # Catch NaN-s
    if np.isnan(benign_score).any():
        logger.warn("WARNING: NaN in benign score list")
        benign_score = [0 if math.isnan(i) else i for i in benign_score]
    if np.isnan(benign_label).any():
        raise ValueError(f"NaN is benign label list\n{benign_label}")
        # benign_label = [0 if math.isnan(i) else i for i in benign_label]
    if np.isnan(malignant_score).any():
        logger.warn("WARNING: NaN in malignant_score list")
        malignant_score = [0 if math.isnan(i) else i for i in malignant_score]
    if np.isnan(malignant_label).any():
        raise ValueError(f"NaN is malignant_label list\n{malignant_label}")
    
    try:
        benign_AUC = sklearn.metrics.roc_auc_score(benign_label, benign_score)
        malignant_AUC = sklearn.metrics.roc_auc_score(malignant_label, malignant_score)

        # Calculate PR AUC for malignant labels
        if auprc:
            precision, recall, _ = sklearn.metrics.precision_recall_curve(malignant_label, malignant_score)
            auprc_res = sklearn.metrics.auc(recall, precision)
    except ValueError as err:
        logger.error(f"Caught error when calculating AUROC: {err}")
        benign_AUC = 0.
        malignant_AUC = 0.

    if report_n is False:
        if auprc:
            return benign_AUC, malignant_AUC, auprc_res
        else:
            return benign_AUC, malignant_AUC
    else:
        return len(keys), benign_AUC, malignant_AUC


def compute_average_AUC_from_dictionary(
    predictions, labels, subgroup_df: pd.DataFrame = None
):
    # AUROC for all patients
    benign_AUC, malignant_AUC, malignant_AUPRC = compute_average_AUC_with_keys(
        predictions, labels, list(predictions.keys()), auprc=True
    )

    # Subgroups BPE
    keys_bpe1 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.bpe == "minimal"].Acc)]
    keys_bpe2 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.bpe == "mild"].Acc)]
    keys_bpe3 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.bpe == "moderate"].Acc)]
    keys_bpe4 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.bpe == "marked"].Acc)]
    keys_bpe_unknown = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.bpe.isin(["TOO MANY", "NOT FOUND"])].Acc)]
    
    # Subgroups age
    keys_under30 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.AgeFix < 30].Acc)]
    keys_under40 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.AgeFix < 40].Acc)]

    # Subgroups BI-RADS
    keys_birads0 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 0].Acc)]
    keys_birads1 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 1].Acc)]
    keys_birads2 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 2].Acc)]
    keys_birads3 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 3].Acc)]
    keys_birads4 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 4].Acc)]
    keys_birads5 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 5].Acc)]
    keys_birads6 = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.birads == 6].Acc)]
    # keys_unknown

    # NME & foci
    keys_nme = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.NME].Acc)]
    keys_foci = [k for k in list(predictions.keys()) if k in list(subgroup_df[subgroup_df.foci].Acc)]


    # Calculate: subgroups BPE
    subgroup_auc = {}
    if len(keys_bpe1) > 0:
        subgroup_auc["bpe1"] = compute_average_AUC_with_keys(predictions, labels, keys_bpe1, report_n=True)
    else:
        subgroup_auc["bpe1"] = (0, 0.0, 0.0)

    if len(keys_bpe2) > 0:
        subgroup_auc["bpe2"] = compute_average_AUC_with_keys(predictions, labels, keys_bpe2, report_n=True)
    else:
        subgroup_auc["bpe2"] = (0, 0.0, 0.0)

    if len(keys_bpe3) > 0:
        subgroup_auc["bpe3"] = compute_average_AUC_with_keys(predictions, labels, keys_bpe3, report_n=True)
    else:
        subgroup_auc["bpe3"] = (0, 0.0, 0.0)

    if len(keys_bpe4) > 0:
        subgroup_auc["bpe4"] = compute_average_AUC_with_keys(predictions, labels, keys_bpe4, report_n=True)
    else:
        subgroup_auc["bpe4"] = (0, 0.0, 0.0)

    if len(keys_bpe_unknown) > 0:
        subgroup_auc["bpe_unknown"] = compute_average_AUC_with_keys(predictions, labels, keys_bpe_unknown, report_n=True)
    else:
        subgroup_auc["bpe_unknown"] = (0, 0.0, 0.0)
    
    # Calculate: subgroups age
    if len(keys_under30) > 0:
        subgroup_auc["age_under30"] = compute_average_AUC_with_keys(predictions, labels, keys_under30, report_n=True)
    else:
        subgroup_auc["age_under30"] = (0, 0.0, 0.0)
    if len(keys_under40) > 0:
        subgroup_auc["age_under40"] = compute_average_AUC_with_keys(predictions, labels, keys_under40, report_n=True)
    else:
        subgroup_auc["age_under40"] = (0, 0.0, 0.0)
    
    # Calculate: subgroups BI-RADS
    if len(keys_birads0) > 0:
        subgroup_auc["birads0"] = compute_average_AUC_with_keys(predictions, labels, keys_birads0, report_n=True)
    else:
        subgroup_auc["birads0"] = (0, 0.0, 0.0)
    if len(keys_birads1) > 0:
        subgroup_auc["birads1"] = compute_average_AUC_with_keys(predictions, labels, keys_birads1, report_n=True)
    else:
        subgroup_auc["birads1"] = (0, 0.0, 0.0)
    if len(keys_birads2) > 0:
        subgroup_auc["birads2"] = compute_average_AUC_with_keys(predictions, labels, keys_birads2, report_n=True)
    else:
        subgroup_auc["birads2"] = (0, 0.0, 0.0)
    if len(keys_birads3) > 0:
        subgroup_auc["birads3"] = compute_average_AUC_with_keys(predictions, labels, keys_birads3, report_n=True)
    else:
        subgroup_auc["birads3"] = (0, 0.0, 0.0)
    if len(keys_birads4) > 0:
        subgroup_auc["birads4"] = compute_average_AUC_with_keys(predictions, labels, keys_birads4, report_n=True)
    else:
        subgroup_auc["birads4"] = (0, 0.0, 0.0)
    if len(keys_birads5) > 0:
        subgroup_auc["birads5"] = compute_average_AUC_with_keys(predictions, labels, keys_birads5, report_n=True)
    else:
        subgroup_auc["birads5"] = (0, 0.0, 0.0)
    if len(keys_birads6) > 0:
        subgroup_auc["birads6"] = compute_average_AUC_with_keys(predictions, labels, keys_birads6, report_n=True)
    else:
        subgroup_auc["birads6"] = (0, 0.0, 0.0)
    
    # Calculate: subgroups w/NME and foci
    if len(keys_nme) > 0:
        subgroup_auc["nme"] = compute_average_AUC_with_keys(predictions, labels, keys_nme, report_n=True)
    else:
        subgroup_auc["nme"] = (0, 0.0, 0.0)
    if len(keys_foci) > 0:
        subgroup_auc["foci"] = compute_average_AUC_with_keys(predictions, labels, keys_foci, report_n=True)
    else:
        subgroup_auc["foci"] = (0, 0.0, 0.0)

    return benign_AUC, malignant_AUC, subgroup_auc, malignant_AUPRC
