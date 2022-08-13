import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, auc, confusion_matrix, roc_curve, precision_recall_curve
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle


# MATPLOTLIB SETTINGS
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "pdf.fonttype": 42
})

font_path = '/helvetica.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.color'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
# plt.rcParams['text.color'] = '#4d4d4d'
# plt.rcParams['xtick.color'] = '#4d4d4d'
# plt.rcParams['ytick.color'] = '#4d4d4d'
# plt.rcParams['axes.labelcolor'] = '#000000'




def leading_zeros_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str



def plot_preds(labels, preds, n_bootstraps=2000, color1hex='#5457A0', color2hex='#8789C0', savefig_name=None, proc=False, decision_threshold=0.02, b=False):
    assert len(labels) == len(preds)
    
    df = pd.DataFrame(data={
        "labels": labels,
        "preds": preds
    })
    
    print(f"{len(df)} pairs of prediction-label provided")
    
    tprs = []
    aurocs = []
    auprcs = []
    precs = []
    recs = []
    base_fpr = np.linspace(0, 1, 101)

    all_labels = []
    all_preds = []
    precision_array = []
    recall_array = np.linspace(0, 1, 100)
    
    senss = []
    specs = []
    ppvs = []
    npvs = []
    
    if df.preds.min() < 0:
        df['preds'] = expit(df.preds)
    
    cm = confusion_matrix(df.labels, df.preds>decision_threshold)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn)
    print("Raw sensitivity:", sensitivity)
    if b is True:
        return

    #plt.figure(figsize=(6, 6))
    for i in trange(n_bootstraps):
        resampled_df = df.sample(n=len(df), replace=True)
        aurocs.append(roc_auc_score(resampled_df.labels, resampled_df.preds))
        fpr, tpr, _ = roc_curve(resampled_df.labels, resampled_df.preds)

        # precision recall
        precision, recall, thresholds = precision_recall_curve(resampled_df.labels, resampled_df.preds)

        precs.append(precision)
        recs.append(recall)
        all_labels.append(resampled_df.labels)
        all_preds.append(resampled_df.preds)
        auprcs.append(auc(recall, precision))

        precision, recall = precision[::-1], recall[::-1]
        prec_array = np.interp(recall_array, recall, precision)
        precision_array.append(prec_array)

        #plt.plot(fpr, tpr, 'b', alpha=0.15)  # uncomment if you want to plot every bootstrap run

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        
        cm = confusion_matrix(resampled_df.labels, resampled_df.preds>decision_threshold)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)

        ppv = tp/(tp+fp)
        npv = tn/(fn+tn)

        senss.append(sensitivity)
        specs.append(specificity)
        ppvs.append(ppv)
        npvs.append(npv)  

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    #mean_prec = np.array(precs).mean(axis=0)
    #mean_rec = np.array(recs).mean(axis=0)

    # get CIs here
    tprs_lower, tprs_upper = np.percentile(tprs, (2.5, 97.5), axis=0)
    auroc_lower, auroc_upper = np.percentile(aurocs, (2.5, 97.5), axis=0)
    auprc_lower, auprc_upper = np.percentile(auprcs, (2.5, 97.5), axis=0)
    sens_lower, sens_upper = np.percentile(senss, (2.5, 97.5), axis=0)
    spec_lower, spec_upper = np.percentile(specs, (2.5, 97.5), axis=0)
    ppv_lower, ppv_upper = np.percentile(ppvs, (2.5, 97.5), axis=0)
    npv_lower, npv_upper = np.percentile(npvs, (2.5, 97.5), axis=0)
    
    mean_auroc = np.array(aurocs).mean(axis=0)
    print(f"AUROC: {mean_auroc} ({auroc_lower}-{auroc_upper})")

    

    # pr
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    #precision_lower, precision_upper = np.percentile(precision_array, (2.5, 97.5), axis=0)
    mean_precision = np.mean(precision_array, axis=0)
    std_precision = np.std(precision_array, axis=0)
    precision_lower, precision_upper = np.percentile(precision_array, (2.5, 97.5), axis=0)
    mean_prauc = auc(recall, precision)
    print(f"AUPRC: {mean_prauc} ({auprc_lower}-{auprc_upper})")
    
    print(f"Sens: {np.mean(senss)} ({sens_lower}-{sens_upper})")
    print(f"spec: {np.mean(specs)} ({spec_lower}-{spec_upper})")
    print(f"ppv: {np.mean(ppvs)} ({ppv_lower}-{ppv_upper})")
    print(f"npv: {np.mean(npvs)} ({npv_lower}-{npv_upper})")

    # ROC plot
#     if proc:
#         fig, axes = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
#     else:
#         fig, axes = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
    
    return_dict = {
        "base_fpr": base_fpr,
        "mean_tprs": mean_tprs,
        "tprs_lower": tprs_lower,
        "tprs_upper": tprs_upper,
        "recall": recall,
        "precision": precision,
        "recall_array": recall_array,
        "precision_lower": precision_lower,
        "precision_upper": precision_upper,
    }
    
    # ROC plot
    plt.figure(figsize=(4,4))
    plt.plot(base_fpr, mean_tprs, color=color1hex, lw=3)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=color2hex, alpha=0.75)
    plt.plot([0, 1], [0, 1],'--',c='grey',alpha=0.5)
    plt.text(0.45, 0.1, f"AUC: {mean_auroc:.3f}\n({auroc_lower:.3f}-{auroc_upper:.3f})")  # print AUC
    ax = plt.gca()
    ax.set_xlim([-0.00, 1.00])
    ax.set_ylim([-0.00, 1.00])
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(leading_zeros_formatter)
    ax.yaxis.set_major_formatter(leading_zeros_formatter)
    if savefig_name:
        plt.savefig(f"rocs_figs/{savefig_name}_roc.pdf", bbox_inches='tight')

    # PR plot
    plt.figure(figsize=(4,4))
    plt.plot(recall, precision, lw=3, alpha=.8, color=color1hex)
    plt.fill_between(recall_array, precision_lower, precision_upper, alpha=0.75, color=color2hex, lw=1)
    plt.text(0.1, 0.1, f"AUC: {mean_prauc:.3f}\n({auprc_lower:.3f}-{auprc_upper:.3f})")  # print AUC
    ax = plt.gca()
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim([-0.00, 1.00])
    ax.set_ylim([-0.00, 1.00])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(leading_zeros_formatter)
    ax.yaxis.set_major_formatter(leading_zeros_formatter)
    if savefig_name:
        plt.savefig(f"rocs_figs/{savefig_name}_prc.pdf", bbox_inches='tight')
    
    # pROC plot
    if proc:      
        plt.figure(figsize=(4,4))
        plt.plot(base_fpr, mean_tprs, color=color1hex, lw=3)
        #plt.text(0.5, 0.1, f"AUC: {mean_auroc:.3f}\n({auroc_lower:.3f}-{auroc_upper:.3f})")  # print AUC

        # Plot partial AUCs
        for idx, x in enumerate(mean_tprs):
            if x>0.9:
                pRoc_thr = idx
                break
        plt.fill_between(base_fpr[:11], np.zeros(shape=len(tprs_lower))[:11], mean_tprs[:11], color='green', alpha=0.3, ls='-', lw=2)  # proc
        plt.fill_between(base_fpr[pRoc_thr:], np.ones(shape=len(tprs_lower))[pRoc_thr:]*0.9, mean_tprs[pRoc_thr:], color='blue', alpha=0.3, ls='-', lw=2)  # proc

        plt.plot([0, 1], [0, 1],'--',c='grey',alpha=0.5)
        ax = plt.gca()
        ax.set_xlim([-0.00, 1.00])
        ax.set_ylim([-0.00, 1.00])
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')
        ax.set_xticks([0, 0.1, 0.5, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
        
        rect1 = Rectangle((0, 0), 0.1, 1.0, color='green', alpha=0.1, lw=2, ls='--')
        rect2 = Rectangle((0, 0.9), 1.0, 0.1, color='blue', alpha=0.1, lw=2, ls='--')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_major_formatter(leading_zeros_formatter)
        ax.yaxis.set_major_formatter(leading_zeros_formatter)
        if savefig_name:
            plt.savefig(f"rocs_figs/{savefig_name}_pauc.pdf", bbox_inches='tight')
            
        return return_dict