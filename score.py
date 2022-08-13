import os
import pickle
import argparse
import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
from scipy.special import expit
import multiprocessing as mp
from tqdm import tqdm
import itertools
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

"""
This script is used to score saved predictions from the inference
In other words, first you need to calculate predictions in the
inference.py script, and then run the output .pkl file through
the score.py
"""


def leading_zeros_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str

def calculate_performance(preds_to_load, preds_lookup: dict, study_level=False):
    # Single model prediction
    if len(preds_to_load) == 1:
        with open(preds_to_load[0], "rb") as f:
            preds = pickle.load(f)
            if type(preds['preds']) == list:
                preds['preds'] = {k[0]: list(v[0]) for k, v in zip(preds['indices'], preds['preds'])}
            if type(preds['labels']) == list:
                preds['labels'] = {k[0]: list(v[0]) for k, v in zip(preds['indices'], preds['labels'])}
    # Ensemble (averaging predictions)
    else:
        preds_array = []
        for p in preds_to_load:
            preds_array.append(preds_lookup[p])

        # Stack all predictions
        combined_preds_dict = {k: [] for k in preds_array[0]['preds'].keys()}
        for p in preds_array:
            p_dict = p['preds']
            for k, v in p_dict.items():
                combined_preds_dict[k].append(v)

        # Average predictions
        averaged_preds_dict = {}
        for k, v in combined_preds_dict.items():
            averaged_preds_dict[k] = np.mean(v, axis=0)

        preds = {
            'indices': preds_array[0]['indices'],
            'labels': preds_array[0]['labels'],
            'preds': averaged_preds_dict
        }
        
    # Convenience arrays for calculations
    labels = np.array(list(preds['labels'].values()))
    logits = np.array(list(preds['preds'].values()))
    indices = [x[0] for x in preds['indices']]

    if not study_level:
        labels_malignant = np.append(labels[:, 1], labels[:, 3])
        logits_malignant = np.append(logits[:, 1], logits[:, 3])

        labels_benign = np.append(labels[:, 0], labels[:, 2])
        logits_benign = np.append(logits[:, 0], logits[:, 2])

        # Compute AUC ROC
        malignant_auroc = roc_auc_score(labels_malignant, logits_malignant)
        try:
            total_auroc = roc_auc_score(labels, logits)
        except:
            total_auroc = 0.0
        try:
            benign_auroc = roc_auc_score(labels_benign, logits_benign)
        except:
            benign_auroc = 0.0

        # Compute AUC PR
        precision, recall, _ = precision_recall_curve(labels_malignant, logits_malignant)
        auprc = auc(recall, precision)
        
        score = malignant_auroc + auprc
    else:
        # Study level

        study_level_labels = []
        for d_ in labels:
            study_level_labels.append((d_[1] or d_[3]))

        study_level_logits_avg = []
        for d_ in logits:
            study_level_logits_avg.append(np.mean((d_[1], d_[3])))

        labels_malignant = study_level_labels
        logits_malignant = study_level_logits_avg
        logits_benign = None
        labels_benign = None
        
        malignant_auroc = roc_auc_score(labels_malignant, logits_malignant)
        benign_auroc = 0.0
        total_auroc = malignant_auroc

        precision, recall, _ = precision_recall_curve(labels_malignant, logits_malignant)
        auprc = auc(recall, precision)
        score = malignant_auroc + auprc
    
    return {
        "preds_to_load": preds_to_load,
        "malignant_auroc": malignant_auroc,
        "benign_auroc": benign_auroc,
        "total_auroc": total_auroc,
        "auprc": auprc,
        "score": score,
        "labels": labels,
        "logits": logits,
        "labels_malignant": labels_malignant,
        "labels_benign": labels_benign,
        "logits_malignant": logits_malignant,
        "logits_benign": logits_benign,
        "indices": indices
    }


def load_preds_from_path(preds_path):
    with open(preds_path, "rb") as f:
        preds = pickle.load(f)
        if type(preds['preds']) == list:
            preds['preds'] = {k[0]: list(v[0]) for k, v in zip(preds['indices'], preds['preds'])}
        if type(preds['labels']) == list:
            preds['labels'] = {k[0]: list(v[0]) for k, v in zip(preds['indices'], preds['labels'])}
    return preds


def bootstrap_ci(labels, logits, decision_threshold, nsamples=2000):
    auc_values = []
    auc_pr_values = []
    senss = []
    specs = []
    ppvs = []
    npvs = []
    
    for b in tqdm(range(nsamples), leave=False):
        idx = np.random.randint(labels.shape[0], size=labels.shape[0])
        try:
            roc_auc = roc_auc_score(labels[idx].ravel(), logits[idx].ravel())
            auc_values.append(roc_auc)
            
            precision, recall, thresholds = precision_recall_curve(labels[idx], logits[idx])
            auc_pr_values.append(auc(recall, precision))
            
            cm = confusion_matrix(labels[idx], logits[idx]>decision_threshold)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)
            
            ppv = tp/(tp+fp)
            try:
                npv = tn/(fn+tn)
            except Exception as e:
                print(e)
                npv = 0
            
            senss.append(sensitivity)
            specs.append(specificity)
            ppvs.append(ppv)
            npvs.append(npv)            
        except ValueError:
            pass
    
    return {
        "auc_roc": np.percentile(auc_values, (2.5, 97.5)),
        "auc_pr": np.percentile(auc_pr_values, (2.5, 97.5)),
        "sens": np.percentile(senss, (2.5, 97.5)),
        "spec": np.percentile(specs, (2.5, 97.5)),
        "ppv": np.percentile(ppvs, (2.5, 97.5)),
        "npv": np.percentile(npvs, (2.5, 97.5)),
    }


def calculate_subgroup_performance(subgroup_df, bootstrap=False,
                                   bootstrap_nsamples=2000, cm_threshold=0.045):
    labels_malignant = np.append(np.array(subgroup_df.labels_lm), np.array(subgroup_df.labels_rm))
    preds_malignant = np.append(np.array(subgroup_df.preds_lm), np.array(subgroup_df.preds_rm))

    labels_benign = np.append(np.array(subgroup_df.labels_lb), np.array(subgroup_df.labels_rb))
    preds_benign = np.append(np.array(subgroup_df.preds_lb), np.array(subgroup_df.preds_rb))
       
    if preds_malignant.min() < 0:  # logits->preds
        preds_malignant = expit(preds_malignant)
        preds_benign = expit(preds_benign)

    # Compute AUC ROC
    malignant_auroc = roc_auc_score(labels_malignant, preds_malignant)
    benign_auroc = roc_auc_score(labels_benign, preds_benign)
    
    if bootstrap:
        ci_dict = bootstrap_ci(labels_malignant, preds_malignant, nsamples=bootstrap_nsamples, decision_threshold=cm_threshold)
    
    malignant_fpr, malignant_tpr, malignant_thresholds = roc_curve(labels_malignant, preds_malignant)
    
    # Compute AUC PR
    precision, recall, thresholds = precision_recall_curve(labels_malignant, preds_malignant)
    auprc = auc(recall, precision)
    
    # Sens, spec, ppv, npv
    cm = confusion_matrix(labels_malignant, preds_malignant>cm_threshold)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    ppv = tp/(tp+fp)
    try:
        npv = tn/(fn+tn)
    except:
        npv = 0
    
    stats_dict = {
        "N": f'{len(subgroup_df):,}',
        "N_all": len(subgroup_df),
        "N_malignant": len(subgroup_df[((subgroup_df.left_malignant==1) | (subgroup_df.right_malignant==1)) & ((subgroup_df.left_benign==0) & (subgroup_df.right_benign==0))]),
        "N_benign": len(subgroup_df[((subgroup_df.left_malignant==0) & (subgroup_df.right_malignant==0)) & ((subgroup_df.left_benign==1) | (subgroup_df.right_benign==1))]),
        "N_malignant_benign": len(subgroup_df[((subgroup_df.left_malignant==1) | (subgroup_df.right_malignant==1)) & ((subgroup_df.left_benign==1) | (subgroup_df.right_benign==1))]),
        "AUROC malignant": round(malignant_auroc, 3),
        "AUROC benign": round(benign_auroc, 3),
        "AUPRC": round(auprc, 3),
        "Sensitivity": round(sensitivity, 3),
        "Specificity": round(specificity, 3),
        "PPV": round(ppv, 3),
        "NPV": round(npv, 3),
        "malignant_fpr": malignant_fpr,
        "malignant_tpr": malignant_tpr,
        "malignant_thresholds": malignant_thresholds,
        "precision": precision,
        "recall": recall
    }
    
    if bootstrap:
        stats_dict['cis'] = ci_dict
    
    return stats_dict


def main(args):
    file_paths = args.i[0]

    preds_lookup = {}

    if not args.save_directory:
        print("You did not provide a directory to save predictions. Your results and plots will *not* be saved.")

    print("Processing the following files:")
    for preds_path in file_paths:
        print(f"\t - {preds_path}")
        preds_lookup[preds_path] = load_preds_from_path(preds_path)

    results = calculate_performance(file_paths, preds_lookup, args.study_level)

    print(f"RESULTS:")
    print("\t Malignant AUROC: ", results['malignant_auroc'])
    print("\t Malignant AUPRC: ", results['auprc'])
    print("\t Benign AUROC: ", results['benign_auroc'])

    if results['logits'].min() < 0:
        results['logits'] = expit(results['logits'])
        results['logits_malignant'] = expit(results['logits_malignant'])
        results['logits_benign'] = expit(results['logits_benign'])
    
    if args.beta_cal_model:
        print("Beta calibration from checkpoint:", args.beta_cal_model)
        estimator = BetaCalibration()
        estimator.load_model(args.beta_cal_model)
        results['logits'][:, 0] = estimator.transform(results['logits'][:, 0])
        results['logits'][:, 1] = estimator.transform(results['logits'][:, 1])
        results['logits'][:, 2] = estimator.transform(results['logits'][:, 2])
        results['logits'][:, 3] = estimator.transform(results['logits'][:, 3])
        results['logits_malignant'] = estimator.transform(results['logits_malignant'])
        results['logits_benign'] = estimator.transform(results['logits_benign'])


    if args.find_best and len(file_paths) > 1:
        all_combinations = []
        for i in range(len(file_paths)):
            all_combinations.extend(list(itertools.combinations(file_paths, i+1)))
        print(f"There are {len(all_combinations)} prediction combinations we will look at.")
        
        partial_calculate = partial(calculate_performance, preds_lookup=preds_lookup, study_level=args.study_level)
        with mp.Pool(processes=40) as p:
            r = list(tqdm(p.imap(partial_calculate, all_combinations), total=len(all_combinations)))
        
        combo_results = max(r, key=lambda item:item['auprc'])
        print(f"BEST COMBO:")
        print(combo_results['preds_to_load'])
        print("\t Malignant AUROC: ", combo_results['malignant_auroc'])
        print("\t Malignant AUPRC: ", combo_results['auprc'])
        print("\t Benign AUROC: ", combo_results['benign_auroc'])
    
    df = pd.DataFrame()
    df['labels'] = results['labels_malignant']
    df['preds'] = results['logits_malignant']
    df['preds_benign'] = results['logits_benign']
    df['indices'] = results['indices'] * 2

    if args.save_directory:
        print("Saving predictions to:", os.path.join(args.save_directory, "data_frame_subgroups.csv"))
        df.to_csv(os.path.join(args.save_directory, "data_frame_subgroups.csv"))
    
    if args.subgroup_analysis:
        subgroup_df = pd.read_pickle("/blinded.pkl")
        subgroup_df_test = subgroup_df[subgroup_df.Acc.isin(results['indices'])]
        subgroup_df_test['Acc'] = pd.Categorical(subgroup_df_test['Acc'], results['indices'])
        subgroup_df_test = subgroup_df_test.sort_values("Acc")

        subgroup_df_test['preds_lb'] = results['logits'][:, 0]
        subgroup_df_test['preds_lm'] = results['logits'][:, 1]
        subgroup_df_test['preds_rb'] = results['logits'][:, 2]
        subgroup_df_test['preds_rm'] = results['logits'][:, 3]

        subgroup_df_test['labels_lb'] = results['labels'][:, 0]
        subgroup_df_test['labels_lm'] = results['labels'][:, 1]
        subgroup_df_test['labels_rb'] = results['labels'][:, 2]
        subgroup_df_test['labels_rm'] = results['labels'][:, 3]

        asian_races = ['Chinese', 'Asian Indian', 'Asian', 'Filipino', 'Asian - unspecified', 'Korean', 'Japanese', 'Other Pacific Islander', 'Thai', 'Laotian', 'Indonesian', 'Vietnamese', 'Pakistani', 'Bangladeshi', 'Hmong', 'Guamanian or Chamorro']

        # Calculate subgroup results
        subgroup_dict = {
            "Overall": calculate_subgroup_performance(subgroup_df_test),
            "BIRADS": {
                "unknown": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.birads.isin(['NO VALID BI-RADS', 'TOO MANY BI-RADS'])], bootstrap=True),   
                "BIRADS 0": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.birads==0], bootstrap=True),
                "BIRADS 6": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.birads==6], bootstrap=True),
                "BIRADS 5": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.birads==5], bootstrap=True),
                "BIRADS 4": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.birads==4], bootstrap=True),
                "BIRADS 1_2_3": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.birads.isin([1, 2, 3])], bootstrap=True),
            },
            "Age": {
                #"Age $<$40": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.AgeFix<40], bootstrap=True),
                "Age $<$50": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.AgeFix<50], bootstrap=True),
                "Age $\geq$50": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.AgeFix>=50], bootstrap=True),
            },
            "Finding": {
                "NME": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.NME]),
                "Foci": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.foci]),
                "Remaining": calculate_subgroup_performance(subgroup_df_test[((~subgroup_df_test.NME) & (~subgroup_df_test.foci))]),
            },
            # "Histological": {
            #     "IDC": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histological=='IDC'], bootstrap=True),  # invasive ductal ca
            #     "DCIS": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histological=='DCIS'], bootstrap=True),  # ductal ca in situ
            #     "ILS": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histological=='ILC'], bootstrap=True), # invasive lobular ca
            #     "Metastatic": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histological=='meta'], bootstrap=True), # invasive lobular ca
            #     "IMC": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histological=='IMC'], bootstrap=True), # invasive lobular ca
            #     "Other": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histological.isin(['IDC', 'DCIS', 'ILS', 'Metastatic', 'IMC'])], bootstrap=True),
            # },
            "Histology": {
                "Other/unknown": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: any(y in str(x) for y in ['unknown', 'lymphocytic lymphoma', 'b-cell lymphoma', 'papillary carcinoma']))], bootstrap=True),
                "IMC": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'IMC' in str(x))], bootstrap=True, subgroup_debug_name="IMC"),
                "Adenoca": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'adenocarcinoma' in str(x))], bootstrap=True),
                # "Papillary": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'papillary carcinoma' in str(x))], bootstrap=True),
                # "Lymphoma": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: any(y in str(x) for y in ['lymphocytic lymphoma', 'b-cell lymphoma']))], bootstrap=True),
                "Meta": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'meta' in str(x))], bootstrap=True),
                "ILC": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'ILC' in str(x))], bootstrap=True),
                "IDC": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'IDC' in str(x))], bootstrap=True),
                "DCIS": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.histology.apply(lambda x: 'DCIS' in str(x))], bootstrap=True, subgroup_debug_name="DCIS"),
            },
            "Molecular": {
                "Triple negative": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.molecular_status=='TN'], bootstrap=True),
                "HER2-enriched": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.molecular_status=='HER2-enriched'], bootstrap=True),
                "Luminal B": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.molecular_status=='luminalB'], bootstrap=True),
                "Luminal A": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.molecular_status=='luminalA'], bootstrap=True),
            },
            "BPE": {
                "Unknown": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.bpe.isin(['NOT FOUND', 'TOO MANY'])], bootstrap=True),
                "Marked": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.bpe=='marked'], bootstrap=True),
                "Moderate": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.bpe=='moderate'], bootstrap=True),
                "Mild": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.bpe=='mild'], bootstrap=True),
                "Minimal": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.bpe=='minimal'], bootstrap=True),
            },
            "Race": {
                #"Other/Unknown": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.firstrace=='Other Race'], bootstrap=True),
                "Other/Unknown": calculate_subgroup_performance(subgroup_df_test[~subgroup_df_test.firstrace.isin(['White', 'African American (Black)'] + asian_races)], bootstrap=True),
                "Black": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.firstrace=='African American (Black)'], bootstrap=True),
                "Asian": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.firstrace.isin(asian_races)], bootstrap=True),
                "White": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.firstrace=='White'], bootstrap=True),
            },
            "Indication": {
                "High-risk screening": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='screening'], bootstrap=True),
                "Other": calculate_subgroup_performance(subgroup_df_test[~subgroup_df_test.indication.isin(['screening', 'unknown'])], bootstrap=True),

                #"extent_of_disease": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='extent_of_disease'], bootstrap=True),
                #"followup_or_surveillance": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='followup_or_surveillance'], bootstrap=True),
                #"workup": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='workup'], bootstrap=True),
                #"other": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication.isin(['conflict', 'implant', 'treatment_response'])], bootstrap=True),
                #"unknown": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='unknown'], bootstrap=True),
                #"Followup": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='followup'], bootstrap=True),
                #"Extent": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='extent_of_disease'], bootstrap=True),
                #"Workup": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.indication=='workup'], bootstrap=True),
            },
            # "Manufacturer": {
            #     "Siemens": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.Manufacturer=='SIEMENS'], bootstrap=True),
            #     "Non-Siemens": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.Manufacturer!='SIEMENS'], bootstrap=True)
            # }
            "Magnet": {
                "1.5T": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.MagnetTeslas==1.5], bootstrap=True),
                "3T": calculate_subgroup_performance(subgroup_df_test[subgroup_df_test.MagnetTeslas==3], bootstrap=True),
            }
        }

        with open("/save.pkl", "wb") as f:
            pickle.dump(subgroup_dict, f)

        
        # Create a dataframe with those subgroup results
        subgroup_results_df = pd.DataFrame(columns=["subgroup", "auroc", "auprc", "sens", "spec"])
        #for c in ['Magnet', 'Age', 'Race', 'BIRADS', 'Molecular', 'Histology', 'Indication', 'BPE']:
        for c in ['Age', 'Race', 'BIRADS', 'Molecular', 'Histology', 'Indication', 'BPE']:
            for sg_name, sg_d in subgroup_dict[c].items():
                subgroup_results_df = subgroup_results_df.append({
                    "category": c,
                    "subgroup": sg_name,
                    "auroc": sg_d['AUROC malignant'],
                    "auprc": sg_d['AUPRC'],
                    "sens": sg_d['Sensitivity'],
                    "spec": sg_d['Specificity'],
                    "N_all": sg_d['N_all'],
                    "N_malignant": sg_d['N_malignant'],
                    "N_benign": sg_d['N_benign'],
                    "N_malignant_benign": sg_d['N_malignant_benign'],
                    "N_negative": sg_d['N_all'] - sg_d['N_malignant_benign'] - sg_d['N_malignant'] - sg_d['N_benign'],
                    "auroc_ci1": sg_d['cis']['auc_roc'][0],
                    "auroc_ci2": sg_d['cis']['auc_roc'][1],
                    "auprc_ci1": sg_d['cis']['auc_pr'][0],
                    "auprc_ci2": sg_d['cis']['auc_pr'][1],
                    "sens_ci1": sg_d['cis']['sens'][0],
                    "sens_ci2": sg_d['cis']['sens'][1],
                    "spec_ci1": sg_d['cis']['spec'][0],
                    "spec_ci2": sg_d['cis']['spec'][1],
                }, ignore_index=True)
        subgroup_results_df['auroc_ci_lower'] = subgroup_results_df.auroc - subgroup_results_df.auroc_ci1
        subgroup_results_df['auroc_ci_upper'] = subgroup_results_df.auroc_ci2 - subgroup_results_df.auroc

        subgroup_results_df['auprc_ci_lower'] = subgroup_results_df.auprc - subgroup_results_df.auprc_ci1
        subgroup_results_df['auprc_ci_upper'] = subgroup_results_df.auprc_ci2 - subgroup_results_df.auprc

        subgroup_results_df['sens_ci_lower'] = subgroup_results_df.sens - subgroup_results_df.sens_ci1
        subgroup_results_df['sens_ci_upper'] = subgroup_results_df.sens_ci2 - subgroup_results_df.sens

        subgroup_results_df['spec_ci_lower'] = subgroup_results_df.spec - subgroup_results_df.spec_ci1
        subgroup_results_df['spec_ci_upper'] = subgroup_results_df.spec_ci2 - subgroup_results_df.spec

        # PLOT subgroup analysis!
        ys = []
        start_y = 0
        space_size = 4  # minimum is 2

        df_for_processing = subgroup_results_df.iloc[::-1]
        for c in ['Age', 'Race', 'BIRADS', 'Molecular', 'Histology', 'Indication', 'BPE']:  # Categories to display on the plot
            df_cat = df_for_processing[df_for_processing.category==c]
            new_ys = np.arange(start_y, (start_y + len(df_cat)), 1)
            ys.extend(new_ys)
            start_y = new_ys[-1] + space_size
            
        reversed_ys = []
        for ix in range(1, len(ys)+1, 1):
            reversed_ys.append(ys[-ix])
        ys = reversed_ys

        plt.rcParams.update({'font.size': 14})
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
        })
        plt.rcParams['pdf.fonttype'] = 42
        font_path = '/helvetica.ttf'  # Your font path goes here
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = prop.get_name()

        fig, axes = plt.subplots(1, 5, figsize=(14,12))
        df_plot_results = subgroup_results_df.iloc[::-1]

        x = df_plot_results.auroc
        y = ys

        xerr_lower = df_plot_results.auroc_ci_lower
        xerr_upper = df_plot_results.auroc_ci_upper

        scatter_size = 75

        axes[0].scatter(x, y, s=scatter_size, zorder=99, facecolor=[(238/256, 118/256, 116/256, 0.5)], edgecolors='black')  # darkorchid
        axes[0].errorbar(x, y, xerr=[xerr_lower, xerr_upper], ls='none', ecolor='#F29492', elinewidth=3)  # plum
        #axes[0].set_yticks(np.arange(len(x)))
        axes[0].set_yticks(ys)
        axes[0].set_yticklabels(df_plot_results.subgroup)
        axes[0].yaxis.grid()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['left'].set_visible(False)
        axes[0].set_xlim([0.6, 1.01])
        axes[0].set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        axes[0].tick_params(left=False) # "for left and bottom ticks"
        axes[0].xaxis.set_major_formatter(leading_zeros_formatter)
        axes[0].set_xlabel('AUC ROC')

        # AUC PR 
        x = df_plot_results.auprc
        axes[1].scatter(x, y, s=scatter_size, zorder=99, facecolor=[(249/256, 181/256, 172/256, 0.5)], edgecolors='black')  # darkslateblue
        axes[1].errorbar(x, y, xerr=[df_plot_results.auprc_ci_lower, df_plot_results.auprc_ci_upper], ls='none', ecolor='#FABBB3', elinewidth=3)  # slateblue
        axes[1].set_yticks(ys)
        axes[1].set_yticklabels([])
        axes[1].tick_params(left=False) # "for left and bottom ticks"
        axes[1].yaxis.grid()
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['left'].set_visible(False)
        axes[1].set_xlim([0.00, 1.04])
        axes[1].set_xticks([0.00, 0.25, 0.5, 0.75, 1.00])
        axes[1].set_xlabel('AUC PR')
        axes[1].xaxis.set_major_formatter(leading_zeros_formatter)

        # Sensitivity
        x = df_plot_results.sens
        axes[2].scatter(x, y, s=scatter_size, zorder=99, facecolor=[(168/256, 199/256, 172/256, 0.5)], edgecolors='black')  #seagreen
        axes[2].errorbar(x, y, xerr=[df_plot_results.sens_ci_lower, df_plot_results.sens_ci_upper], ls='none', ecolor='#B4CFB8', elinewidth=3)  #mediumseagreen
        axes[2].set_yticks(ys)
        axes[2].set_yticklabels([])
        axes[2].tick_params(left=False) # "for left and bottom ticks"
        axes[2].yaxis.grid()
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[2].spines['left'].set_visible(False)
        #axes[2].set_xlim([0.00, 1.04])
        #axes[2].set_xticks([0.00, 0.25, 0.5, 0.75, 1.00])
        axes[2].set_xlim([0.70, 1.02])
        axes[2].set_xticks([0.7, 0.8, 0.9, 1.00])
        axes[2].set_xlabel('Sensitivity')
        axes[2].xaxis.set_major_formatter(leading_zeros_formatter)

        # Specificity
        x = df_plot_results.spec
        axes[3].scatter(x, y, s=scatter_size, zorder=99, facecolor=[(208/256, 214/256, 181/256, 0.5)], edgecolors='black')  #skyblue
        axes[3].errorbar(x, y, xerr=[df_plot_results.spec_ci_lower, df_plot_results.spec_ci_upper], ls='none', ecolor='#ADB77B', elinewidth=3)  # lightskyblue
        axes[3].set_yticks(ys)
        axes[3].set_yticklabels([])
        axes[3].tick_params(left=False) # "for left and bottom ticks"
        axes[3].yaxis.grid()
        axes[3].spines['top'].set_visible(False)
        axes[3].spines['right'].set_visible(False)
        axes[3].spines['left'].set_visible(False)
        #axes[3].set_xlim([0.00, 1.04])
        #axes[3].set_xticks([0.00, 0.25, 0.5, 0.75, 1.00])
        axes[3].set_xlim([0.25, 1.00])
        axes[3].set_xticks([0.25, 0.5, 0.75, 1.00])
        axes[3].set_xlabel('Specificity')
        axes[3].xaxis.set_major_formatter(leading_zeros_formatter)

        # Malignant/benign
        axes[4].barh(
            y,
            (df_plot_results.N_malignant + df_plot_results.N_malignant_benign),
            align='center',
            color='firebrick'
        )
        axes[4].barh(
            y,
            (df_plot_results.N_benign + df_plot_results.N_negative),
            left=(df_plot_results.N_malignant + df_plot_results.N_malignant_benign),
            align='center',
            color='yellowgreen'
        )
        axes[4].spines['top'].set_visible(False)
        axes[4].spines['right'].set_visible(False)
        axes[4].spines['left'].set_visible(False)
        axes[4].set_yticks(ys)
        axes[4].set_yticklabels([])
        #axes[4].set_xscale('log')
        axes[4].set_xscale('linear')
        axes[4].set_xlabel('N malignant / non-mal.')

        if args.save_directory:
            plt.savefig(os.path.join(args.save_directory, 'subgroup_analysis.pdf'))  # SAVE FIGURE (SUBGROUP ANALYSIS)
            plt.savefig(os.path.join(args.save_directory, 'subgroup_analysis.png'))

        # PRINT LATEX-FORMATTED TABLE WITH RESULTS
        print("Group & N & AUROC & AUPRC & Sens & Spec & PPV & NPV \\\\")
        for group, gd in subgroup_dict.items():
            display_groups = ['BIRADS', 'Age', 'Histology', 'Molecular', 'BPE', 'Race', 'Indication']  # those groups will get displayes
            dont_display_groups = ['Finding']  # those will not
            if group in display_groups:
                print(f"{group} & & & & & & & \\\\")
                for i, v in gd.items():
                    print("$\qquad$ {} & {} & {:.2f} ({:.2f}-{:.2f}) & {:.2f} ({:.2f}-{:.2f}) & {:.2f} ({:.2f}-{:.2f}) & {:.2f} ({:.2f}-{:.2f}) & {:.2f} ({:.2f}-{:.2f}) & {:.2f} ({:.2f}-{:.2f}) \\\\".format(
                        i, v['N'], 
                        v['AUROC malignant'], v['cis']['auc_roc'][0], v['cis']['auc_roc'][1], 
                        v['AUPRC'], v['cis']['auc_pr'][0], v['cis']['auc_pr'][1], 
                        v['Sensitivity'], v['cis']['sens'][0], v['cis']['sens'][1], 
                        v['Specificity'], v['cis']['spec'][0], v['cis']['spec'][1], 
                        v['PPV'], v['cis']['ppv'][0], v['cis']['ppv'][1], 
                        v['NPV'], v['cis']['npv'][0], v['cis']['npv'][1], 
                    ))


        # PLOT ROC CURVES FOR SUBGROUPS
        plot_dictionary = {
            "BIRADS": ["BIRADS 0", "BIRADS 1_2_3", "BIRADS 4", "BIRADS 5", "BIRADS 6"],
            "BPE": ["Minimal", "Mild", "Moderate", "Marked", "Unknown"],
            "Histology": ["IDC", "DCIS", "ILC", "Meta", "IMC", "Adenoca", "Other/unknown"],
            "Molecular": ["Luminal A", "Luminal B", "Triple negative", "HER2-enriched"],
            "Race": ["White", "Black", "Asian", "Other/Unknown"],
            #"Magnet": ["1.5T", "3T"]
            "Indication": ["High-risk screening", "Other"]
        }

        
        plt.rcParams.update({'font.size': 20})
        for plot_group, plot_subgroups in plot_dictionary.items():
            plt.figure(figsize=(6,6))
            for plot_subgroup in plot_subgroups:
                plt.plot(
                    subgroup_dict[plot_group][plot_subgroup]['malignant_fpr'],
                    subgroup_dict[plot_group][plot_subgroup]['malignant_tpr'],
                    label=plot_subgroup,
                    linewidth=3
                )
            plt.plot( [0,1],[0,1], '--', c='grey', alpha=0.5)  #identity line
            plt.legend(loc='lower right')
            plt.ylim(0, 1.01)
            plt.xlim(-0.005, 1.01)
            #plt.title(f"ROC curves in subgroups: {plot_group}")
            plt.ylabel("Sensitivity")
            plt.xlabel("1 - Specificity")
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_major_formatter(leading_zeros_formatter)
            ax.yaxis.set_major_formatter(leading_zeros_formatter)
            if args.save_directory:
                plt.savefig(os.path.join(args.save_directory, f'subgroup_{plot_group}.pdf'))
                plt.savefig(os.path.join(args.save_directory, f'subgroup_{plot_group}.png'))


            # Precision-recall plot
            plt.figure(figsize=(6,6))
            for plot_subgroup in plot_subgroups:
                plt.plot(
                    subgroup_dict[plot_group][plot_subgroup]['recall'],
                    subgroup_dict[plot_group][plot_subgroup]['precision'],
                    label=plot_subgroup,
                    linewidth=3
                )
            plt.legend(loc='lower left')
            plt.ylim(0, 1.00)
            plt.xlim(0, 1.00)
            #plt.title(f"PR curves in subgroups: {plot_group}")
            plt.ylabel("Precision")
            plt.xlabel("Recall")
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_major_formatter(leading_zeros_formatter)
            ax.yaxis.set_major_formatter(leading_zeros_formatter)
            if args.save_directory:
                plt.savefig(os.path.join(args.save_directory, f'subgroup_{plot_group}_PR.pdf'))
                plt.savefig(os.path.join(args.save_directory, f'subgroup_{plot_group}_PR.png'))

        print("\nSubgroup analysis complete")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MRI Scoring")
    parser.add_argument("-i", action='append', nargs='*')
    parser.add_argument("--find_best", default=False, type=bool, help='If multiple prediction files are loaded, the script will attempt to find the best combination.')
    parser.add_argument("--save_directory", type=str, help='A DIRECTORY where results should be saved')
    parser.add_argument("--subgroup_analysis", default=False, type=bool)
    parser.add_argument("--study_level", default=False, type=bool)
    parser.add_argument("--beta_cal_model", help='If you have beta calibration model saved, provide path here and it will be used to calibrate predictions')
    args = parser.parse_args()

    main(args)
