import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchio
from dataset import MRIDataset
from apex import amp
from networks import resnet_3d_pre_post
from networks.models import MultiSequenceChannels
from tqdm import tqdm
from utilities.utils import boolean_string
import random
from sklearn.metrics import roc_auc_score


def tta_compose(matrix: torch.Tensor, torchio_compose: torchio.transforms.Compose):
    """
    Performs test-time augmentation on a matrix
    This function handles multiple batches, so that
    If `matrix` is ndim=5, then it is expected to have
    shape {B, C, X, Y, Z} where:
    B - batch; C - channel; X, Y, Z - volume dims.
    If ndim=4, then it should have {C, X, Y, Z} shape.

    It *always* returns a tensor of shape {B, C, X, Y, Z}, i.e.
    ready to be inputted into the network
    """
    transformed_tensors = []

    if matrix.ndim == 5:
        for datum in matrix:
            datum_trans = torchio_compose(datum)
            transformed_tensors.append(datum_trans)
    elif matrix.ndim == 4:
        datum_trans = torchio_compose(matrix)
        transformed_tensors.append(datum_trans)
    else:
        raise NotImplementedError(f"Input ndim={matrix.ndim} must be 4 or 5")
    
    matrix_trans = torch.stack(transformed_tensors)

    return matrix_trans


def interim_scores(preds, indices, labels):
    # Single model prediction
    if type(preds) == list:
        preds = {k[0]: list(v[0]) for k, v in zip(indices, preds)}
    if type(labels) == list:
        labels = {k[0]: list(v[0]) for k, v in zip(indices, labels)}
        
    # Convenience arrays for calculations
    labels = np.array(list(labels.values()))
    logits = np.array(list(preds.values()))
    indices = [x[0] for x in indices]

    labels_malignant = np.append(labels[:, 1], labels[:, 3])
    logits_malignant = np.append(logits[:, 1], logits[:, 3])

    # Compute AUC ROC
    try:
        auroc = roc_auc_score(labels_malignant, logits_malignant)
    except:
        auroc = 0.0
    return auroc



def main(base_parameters):
    print("Inference")

    torch.manual_seed(base_parameters.seed)
    torch.cuda.manual_seed(base_parameters.seed)
    np.random.seed(base_parameters.seed)

    # Set up model, optimizer
    if base_parameters.architecture == 'multi_channel':
        if base_parameters.age_as_channel:
            in_channels = 4
        else:
            in_channels = 3
        model = MultiSequenceChannels(
            base_parameters,
            in_channels=in_channels,
            inplanes=base_parameters.inplanes,
            return_hidden=base_parameters.save_last_hidden
        )
        print("Loaded multi channel")
    else:
        model = resnet_3d_pre_post.MRIResNet3D_wls_right(
            resnet_size=18,
            groups=base_parameters.resnet_groups,
            in_channels=1,
            topk=base_parameters.topk,
            return_h=base_parameters.save_last_hidden
        ).float()
    model.to("cuda")

    # Load weights
    weights_path = base_parameters.weights
    checkpoint = torch.load(weights_path)
    #checkpoint_amp = torch.load(base_parameters.weightsAmp)
    
    # if base_parameters.architecture == 'multi_channel':
    #     fixed_weights = {}
    #     for k, v in checkpoint['model'].items():
    #         fixed_weights[(k.replace("feature_extractor.", ""))] = checkpoint['model'][k]
    #     checkpoint['model'] = fixed_weights
    # THIS IS FOR MODELS TRAINED WITH DDP / MULTIGPU
    if base_parameters.trained_with_ddp:
        print("Loading weights...")
        fixed_weights = {}
        for k, v in checkpoint['model'].items():
            fixed_weights[(k.replace("module.", ""))] = checkpoint['model'][k]
        checkpoint['model'] = fixed_weights
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_parameters.lr, weight_decay=1e-4)
    model, optimizer = amp.initialize(model, optimizer, opt_level=base_parameters.half_level, min_loss_scale=128)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'amp' in checkpoint:
        amp.load_state_dict(checkpoint['amp'])

    model.eval()
    

    # Dataset
    val_dataset = MRIDataset(
        base_parameters,
        base_parameters.subset
    )
    print("Loaded dataset of length:", len(val_dataset))

    # Data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_parameters.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=base_parameters.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Inference and collect outputs
    all_preds = []
    all_indices = []
    all_labels = []
    all_hidden = []
    validation_predictions = dict()
    validation_labels = val_dataset.get_labels()

    # TTA information output
    if base_parameters.tta_rounds > 0:
        print("Selected TTA:")
        print(f"*** ({base_parameters.tta_rounds} rounds)")
        if base_parameters.tta_flip:
            print("*** random horizontal flip")
        print("*** random affine augmentations")
    else:
        print("No test-time augmentations will be performed.")

    # Inference loop
    interim_auroc = 0.0
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i_batch, batch in pbar:
        indices, raw_data_batch, label_batch, _ = batch
        label_batch = label_batch.to("cuda")
        pbar.set_description(f"Batch {i_batch} *** interim AUROC: {interim_auroc}")
        with torch.no_grad():
            # Prediction without augmentation
            if base_parameters.save_last_hidden:
                preds, h = model(raw_data_batch.to("cuda"), return_logits=base_parameters.return_logits)
            else:
                preds = model(raw_data_batch.to("cuda"), return_logits=base_parameters.return_logits)
                        
            # Test-time augmentations
            # perform augmentation N times
            # for each augmented volume generate predictions
            # average N predictions
            if base_parameters.tta_rounds > 0:
                for round_i in range(base_parameters.tta_rounds):
                    pbar.set_description(f"Batch {i_batch} [TTA #{round_i+1}] *** interim AUROC: {interim_auroc}")
                    # Randomly choose whether to do horizontal flip
                    if base_parameters.tta_flip:
                        if random.random() > 0.5:
                            flip_probability=1.0
                        else:
                            flip_probability=0.0
                    else:
                        flip_probability=0.0
                    
                    if base_parameters.tta_gamma:
                        if random.random() > 0.5:
                            gamma_probability=1.0
                        else:
                            gamma_probability=0.0
                    else:
                        gamma_probability=0.0
                    
                    if base_parameters.tta_blur:
                        if random.random() > 0.5:
                            blur_probability=1.0
                        else:
                            blur_probability=0.0
                    else:
                        blur_probability=0.0
                    

                    # Compose all TTA augmentations
                    torchio_compose = torchio.transforms.Compose([
                        torchio.transforms.RandomFlip(axes=2, flip_probability=flip_probability),
                        torchio.transforms.RandomAffine(
                            scales=base_parameters.tta_affine_scales,
                            degrees=base_parameters.tta_affine_degrees,
                            translation=base_parameters.tta_affine_translation,
                            p=0.5
                        ),
                        torchio.transforms.RandomGamma(p=gamma_probability),
                        torchio.transforms.RandomBlur(p=blur_probability)
                    ])
                    
                    # Augment the volume
                    if base_parameters.architecture == 'multi_channel':
                        augmented_volume = tta_compose(raw_data_batch, torchio_compose)
                    else:
                        augmented_volume = tta_compose(raw_data_batch, torchio_compose)
                    
                    # Generate predictions
                    preds_tta_ = model(
                        augmented_volume.to("cuda"),
                        return_logits=base_parameters.return_logits
                    )
                    
                    # If horizontal flip was applied, flip predictions now
                    if flip_probability == 1.0:
                        preds_tta = torch.tensor([[preds_tta_[0][2], preds_tta_[0][3], preds_tta_[0][0], preds_tta_[0][1]]], device='cuda')
                    else:
                        preds_tta = preds_tta_
                    
                    # Add to predictions
                    preds += preds_tta
                
                preds /= (base_parameters.tta_rounds + 1)

            for i in range(0, len(label_batch)):
                validation_predictions[indices[i]] = preds[i].cpu().detach().numpy()
            
            all_preds.append(preds.detach().cpu().numpy())
            all_indices.append(indices)
            all_labels.append(label_batch.detach().cpu().numpy())

            if base_parameters.save_last_hidden:
                all_hidden.append(h.detach().cpu().numpy())
            
            interim_auroc = interim_scores(all_preds, all_indices, all_labels)
    
    # Save to pickle
    print(f"Saving to pickle: {base_parameters.output}")
    output = {
        "weights": weights_path,
        "preds": validation_predictions,
        "indices": all_indices,
        "labels": validation_labels,
        "hidden": all_hidden
    }
    with open(base_parameters.output, "wb") as f:
        pickle.dump(output, f)
    return


def get_args():
    parser = argparse.ArgumentParser("MRI Inference")

    parser.add_argument("-m", "--metadata", type=str, default="/blinded.pkl")
    parser.add_argument("-d", "--datalist", type=str, default="/blinded.pkl")

    parser.add_argument("-s", "--subgroup", type=str, default="/blinded.pkl")
    parser.add_argument("-o", "--output", type=str, required=True, help='Path to pkl file where outputs should be saved')
    parser.add_argument("-w", "--weights", type=str, required=True, help='Path to weights')
    parser.add_argument("--subset", type=str, default='validation')
    parser.add_argument("--save_last_hidden", type=boolean_string, default=False, help='Save last hidden representation')
    
    parser.add_argument("--validation_fraction", type=float, default=1.00)

    parser.add_argument("--architecture", type=str, default="3d_resnet18")
    parser.add_argument("--resnet_groups", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=19)
    parser.add_argument("--input_type", type=str, default="sub_t1c2")
    parser.add_argument("--input_size", type=str, default="normal")
    parser.add_argument("--label_type", type=str, default="cancer")
    parser.add_argument("--aug_policy", type=str, default="none")
    parser.add_argument("--cutout", type=boolean_string, default=False)
    parser.add_argument("--age_as_channel", type=boolean_string, default=False)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--subtraction_clipping", type=boolean_string, default=False)
    parser.add_argument("--mixup", type=boolean_string, default=False)
    parser.add_argument("--isotropic", type=boolean_string, default=False)
    parser.add_argument("--inplanes", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--half_level", type=str, default='O2', choices={'O1', 'O2'})
    parser.add_argument("--trained_with_ddp", type=boolean_string, default=False)

    parser.add_argument("--tcia_duke", type=bool, default=False, help='Use this flag when using Duke data')
    parser.add_argument("--tcga_brca", type=bool, default=False, help='Use this flag when using TCGA data OR UJ data')

    # test-time augmentations
    parser.add_argument("--tta_rounds", type=int, default=0, help='How many rounds of TTA per image')
    parser.add_argument("--tta_flip", type=boolean_string, default=False)
    parser.add_argument("--tta_gamma", type=boolean_string, default=False)
    parser.add_argument("--tta_blur", type=boolean_string, default=False)
    parser.add_argument("--tta_affine_scales", type=float, default=0.1)
    parser.add_argument("--tta_affine_degrees", type=int, default=10)
    parser.add_argument("--tta_affine_translation", type=int, default=10)

    # deprecated tta
    parser.add_argument("--tta_zoomout", type=boolean_string, default=False)
    parser.add_argument("--tta_noise", type=boolean_string, default=False)
    parser.add_argument("--tta_ghosting", type=boolean_string, default=False)

    
    parser.add_argument("--return_logits", type=boolean_string, default=False)

    args = parser.parse_args()

    # Load data for subgroup statistics
    subgroup_df = pd.read_pickle(args.subgroup)
    args.subgroup_df = subgroup_df

    return args


if __name__ == "__main__":
    parameters = get_args()
    main(parameters)
