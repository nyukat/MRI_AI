import os
import numpy as np
import math
import pickle
import random
from random import randrange

from utilities import pickling
import torch
from torch.utils.data import Dataset
import torchvision
import collections
import SimpleITK as sitk
import h5py
import torchio
from argparse import Namespace
from typing import List



def resample_volume(volume: sitk.Image,
                    new_spacing: List[float],
                    interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    Change spacing
    :param volume: Input image to be resampled
    :param new_spacing: For our volumes this should be a 3-item list
                        i.e. (1.0, 1.0, 1.0)
    :param interpolator: Which interpolator to use
    """
    new_size = [int(round(osz*ospc/nspc)) 
                for osz, ospc, nspc 
                in zip(volume.GetSize(), volume.GetSpacing(), new_spacing)]
    minimum_value = sitk.GetArrayFromImage(volume).min()
    return sitk.Resample(volume,
                         new_size,
                         sitk.Transform(),
                         interpolator,
                         volume.GetOrigin(),
                         new_spacing,
                         volume.GetDirection(),
                         int(minimum_value),
                         volume.GetPixelID())



class MRIDataset(Dataset):
    def __init__(self, parameters, phase: str):
        if type(parameters) == dict:
            parameters = Namespace(**parameters)
        self.parameters = parameters
        self.phase = phase
        self.random_seed = 0
        metadata = pickle.load(open(parameters.metadata, "rb"))
        self.metadata = metadata

        assert phase in ["training", "validation", "test"], "Unspecified phase"

        # Read data list
        (
            data_list_training,
            data_list_validation,
            data_list_test,
        ) = pickling.unpickle_from_file(self.parameters.datalist)
        print(parameters.label_type)
        print("data list length before:", len(data_list_training), len(data_list_validation), len(data_list_test))

        if parameters.label_type == 'birads':
            # only birads 1,2,3,4,5,6
            print("Generating data list for BIRADS")
            good_birads = parameters.subgroup_df[parameters.subgroup_df.birads.isin([1,2,3,4,5,6])]
            good_birads_acns = good_birads.Acc.values
            data_list_training = {k: v for k, v in data_list_training.items() if k in good_birads_acns}
            data_list_validation = {k: v for k, v in data_list_validation.items() if k in good_birads_acns}
            
            # Convert BI-RADS to one-hot labels
            # Combine 4+5 as one class
            birads_dict = {}
            birads_dict_temp = good_birads[['Acc', 'birads']].set_index('Acc').to_dict(orient='index')
            for acc, d in birads_dict_temp.items():
                b = d['birads']  # This study bi-rads
                new_label = np.zeros((5), dtype=int)
                if b == 1:  # bi-rads 1
                    new_label[0] = 1
                elif b == 2:  # bi-rads 2
                    new_label[1] = 1
                elif b == 3:  # bi-rads 3
                    new_label[2] = 1
                elif b == 4 or b == 5:  # bi-rads 4 and 5
                    new_label[3] = 1
                elif b == 6:  # bi-rads 6
                    new_label[4] = 1
                
                birads_dict[acc] = new_label
            self.birads_dict = birads_dict
        elif parameters.label_type == 'bpe':
            # Only studies with known BPE
            print("Generating data list for BPE")
            good_bpe = parameters.subgroup_df[parameters.subgroup_df.bpe.isin(['minimal', 'mild', 'moderate', 'marked'])]
            good_bpe_acns = good_bpe.Acc.values
            data_list_training = {k: v for k, v in data_list_training.items() if k in good_bpe_acns}
            data_list_validation = {k: v for k, v in data_list_validation.items() if k in good_bpe_acns}
            bpe_labels_dict = {}
            bpe_labels_temporary = good_bpe[['Acc', 'bpe']].set_index('Acc').to_dict(orient='index')
            for acc, d in bpe_labels_temporary.items():
                bpe = d['bpe']
                new_label = np.zeros((4), dtype=int)
                if bpe == 'minimal':
                    new_label[0] = 1
                elif bpe == 'mild':
                    new_label[1] = 1
                elif bpe == 'moderate':
                    new_label[2] = 1
                elif bpe == 'marked':
                    new_label[3] = 1
                bpe_labels_dict[acc] = new_label
            self.bpe_labels_dict = bpe_labels_dict
        print("data list length after:", len(data_list_training), len(data_list_validation), len(data_list_test))
        
        if phase == "training":
            self.data_list = data_list_training
            number_of_all_training_examples = len(self.data_list)
            if self.parameters.number_of_training_samples is not None:
                number_of_remaining_training_examples = self.parameters.number_of_training_samples
            else:
                number_of_remaining_training_examples = math.ceil(
                    number_of_all_training_examples * self.parameters.training_fraction
                )
            print(f"*** There are {number_of_remaining_training_examples} training examples to go through")
            self.data_list = collections.OrderedDict(
                list(self.data_list.items())[:number_of_remaining_training_examples]
            )
        elif phase == "validation":
            if self.parameters.validation_fraction != 1. or self.parameters.number_of_validation_samples is not None:
                if self.parameters.number_of_validation_samples is not None:
                    num_val_examples = self.parameters.number_of_validation_samples
                else:
                    num_val_examples = math.ceil(
                        len(data_list_validation) * self.parameters.validation_fraction
                    )
                print(f"*** There are {num_val_examples} val examples to go through")
                self.data_list = collections.OrderedDict(
                    list(data_list_validation.items())[:num_val_examples]
                )
            else:
                self.data_list = collections.OrderedDict(data_list_validation)
        elif phase == "test":
            self.data_list = collections.OrderedDict(data_list_test)
        
        self.add_metadata_to_datalist()

        self.exam_list = list(self.data_list.keys())
    
    def add_metadata_to_datalist(self):
        # Adds metadata about patient age, BPE, BI-RADS to the data list
        df = self.parameters.subgroup_df
        for acn in self.data_list.keys():
            try:
                df_acn = df[df.Acc==acn]
                if len(df_acn) == 1:
                    self.data_list[acn]['bpe'] = df_acn.bpe.values[0]
                    self.data_list[acn]['birads'] = df_acn.birads.values[0]
                    self.data_list[acn]['age'] = df_acn.AgeFix.values[0]
            except Exception as why:
                print(f"Failed fetching metadata for acc {acn}: {why}")
    
    def h5_to_sitk(self, h5path):
        """
        Load h5 image and return SimpleITK Image
        """
        
        acn = os.path.dirname(h5path).split("/")[-1]
        
        for pt_counter, (mrn, patient_metadata) in enumerate(self.metadata.items()):
            studies = patient_metadata["ExamsInfo"]
            for study in studies:
                try:
                    int_acn = int(acn)
                except:
                    int_acn = acn
                if study["AccessionNumber"].strip() == acn or study["AccessionNumber"].strip() == int_acn:
                    series = study["SeriesInfo"]
                    for serie in series:
                        if os.path.basename(h5path) == os.path.basename(serie["ImagePath"]) \
                        and os.path.basename(os.path.dirname(h5path)) == os.path.basename(os.path.dirname(serie['ImagePath'])):
                            spacing = serie['PixelSpacing']
                            cosines = serie['Cosines']
                            
                            # Load file
                            f = h5py.File(h5path, 'r')
                            image = f['data']
                            image = np.array(f['data']).astype(np.float32)
                            
                            # Generate ITK image
                            itkimage = sitk.GetImageFromArray(image, isVector=False)
                            itkimage.SetSpacing(spacing)
                            itkimage.SetDirection(cosines)

                            # If requested, resample to isotropic voxels
                            if self.parameters.isotropic:
                                if itkimage.GetSpacing() == (1.0, 1.0, 1.0):
                                    pass  # already isotropic
                                else:
                                    itkimage = resample_volume(
                                        volume=itkimage,
                                        new_spacing=[1.0, 1.0, 1.0]
                                    )
                            
                            return itkimage

    def get_datum(self, index):
        item_key = self.exam_list[index]
        datalist = self.data_list[item_key]

        if self.parameters.label_type == 'birads':
            label = self.birads_dict[item_key]
            label = torch.tensor(label).long()
        elif self.parameters.label_type == 'bpe':
            label = self.bpe_labels_dict[item_key]
            label = torch.tensor(label).long()
        else:
            # Default - cancer label
            label = torch.Tensor(datalist["label"]).long()

        # Read images depending on which input is used
        if self.parameters.input_type == 'random':
            if self.phase == "training":
                arr_input_types = ['sub_t1c1', 'sub_t1c2', 't1pre', 't1c1', 't1c2']
                self.parameters.input_type = arr_input_types[torch.randint(low=0, high=len(arr_input_types), size=(1,))[0]]

        affine = None  # affine matrix
        if self.parameters.input_type in ['sub_t1c1', 'sub_t1c2', 't1pre', 'three_channel', 'five_channel', 'random', 'multi']:
            item_pre = self.h5_to_sitk(datalist["image"]["pre"], datalist)
            affine = affine_from_sitk(item_pre)
        if self.parameters.input_type in ['sub_t1c1', 't1c1', 'three_channel', 'five_channel', 'random', 'multi', 'MIL']:
            item_post1 = self.h5_to_sitk(datalist["image"]["post1"], datalist)
            if affine is None:
                affine = affine_from_sitk(item_post1)
        if self.parameters.input_type in ['sub_t1c2', 't1c2', 'mip_t1c2', 'three_channel', 'five_channel', 'random', 'multi']:
            item_post2 = self.h5_to_sitk(datalist["image"]["post2"], datalist)
            if affine is None:
                affine = affine_from_sitk(item_post2)
        if self.parameters.input_type in ['t2', 'MIL']:
            if 't2' in datalist['image']:
                item_t2 = self.h5_to_sitk(datalist["image"]["t2"], datalist)
                if affine is None:
                    affine = affine_from_sitk(item_t2)
            else:
                item_t2 = None
                    
        # SimpleITK -> numpy 3D matrices -> 4D matrices
        if self.parameters.input_type in ['sub_t1c1', 'sub_t1c2', 't1pre', 'three_channel', 'five_channel', 'random', 'multi']:
            item_pre = np.expand_dims(sitk.GetArrayFromImage(item_pre), 0)
        if self.parameters.input_type in ['sub_t1c1', 't1c1', 'three_channel', 'five_channel', 'random', 'multi', 'MIL']:
            item_post1 = np.expand_dims(sitk.GetArrayFromImage(item_post1), 0)
        if self.parameters.input_type in ['sub_t1c2', 't1c2', 'mip_t1c2', 'three_channel', 'five_channel', 'random', 'multi']:
            item_post2 = np.expand_dims(sitk.GetArrayFromImage(item_post2), 0)
        if self.parameters.input_type in ['t2', 'MIL']:
            if item_t2 is not None:
                item_t2 = np.expand_dims(sitk.GetArrayFromImage(item_t2), 0)

        # Create torchio (tio) Image objects required for augmentations
        if self.parameters.input_type in ['sub_t1c1', 'sub_t1c2', 't1pre', 'three_channel', 'five_channel', 'random', 'multi']:
            tio_pre = torchio.ScalarImage(tensor=item_pre, affine=affine)
        if self.parameters.input_type in ['sub_t1c1', 't1c1', 'three_channel', 'five_channel', 'random', 'multi', 'MIL']:
            tio_post1 = torchio.ScalarImage(tensor=item_post1, affine=affine)
        if self.parameters.input_type in ['sub_t1c2', 't1c2', 'three_channel', 'five_channel', 'random', 'multi']:
            tio_post2 = torchio.ScalarImage(tensor=item_post2, affine=affine)
        if self.parameters.input_type == 'mip_t1c2':
            mip = np.expand_dims(np.amax(item_post2, axis=1), 0)
        if self.parameters.input_type in ['t2', 'MIL']:
            if item_t2 is not None:
                tio_t2 = torchio.ScalarImage(tensor=item_t2, affine=affine)

        # Create subject
        if self.parameters.input_type == 'sub_t1c1':
            subtraction = (tio_post1.data - tio_pre.data)
            del tio_post1
            del tio_pre
            if self.parameters.subtraction_clipping:
                subtraction = torch.clamp(subtraction, 0, subtraction.max())
            tio_subtraction = torchio.ScalarImage(
                tensor=subtraction,
                affine=affine
            )
            subject = torchio.Subject({"sub_t1c1": tio_subtraction})
        elif self.parameters.input_type == 'sub_t1c2':
            subtraction = (tio_post2.data - tio_pre.data)
            del tio_post2
            del tio_pre
            if self.parameters.subtraction_clipping:
                subtraction = torch.clamp(subtraction, 0, subtraction.max())
            tio_subtraction = torchio.ScalarImage(
                tensor=subtraction,
                affine=affine
            )
            subject = torchio.Subject({"sub_t1c2": tio_subtraction})
        elif self.parameters.input_type == 't1c1':
            subject = torchio.Subject({"post1": tio_post1})
        elif self.parameters.input_type == 't1c2':
            subject = torchio.Subject({"post2": tio_post2})
        elif self.parameters.input_type == 't1pre':
            subject = torchio.Subject({"pre": tio_pre})
        elif self.parameters.input_type in ['three_channel', 'five_channel']:
            subject = torchio.Subject({"pre": tio_pre, "post1": tio_post1, "post2": tio_post2})
        elif self.parameters.input_type == 'mip_t1c2':
            subject = torchio.ScalarImage(tensor=mip)
            #print("*** Subject shape:", subject.shape)
        elif self.parameters.input_type == 't2':
            subject = torchio.Subject({"t2": tio_t2})
        elif self.parameters.input_type in ['random', 'multi']:
            subject = torchio.Subject({
                "pre": tio_pre,
                "post1": tio_post1,
                "post2": tio_post2,
            })
        elif self.parameters.input_type == 'MIL':
            # crop before making Subject, sizes must match
            crop_or_pad = torchio.transforms.CropOrPad((190,448,448))

            tio_post1 = crop_or_pad(tio_post1)
            if item_t2 is not None:
                tio_t2 = crop_or_pad(tio_t2)
                subject = torchio.Subject({
                    "sub_t1c1": tio_post1,
                    "t2": tio_t2
                })
            else:
                subject = torchio.Subject({"sub_t1c1": tio_post1})
        
        # Crop/Pad
        if self.parameters.input_type == 'mip_t1c2':
            pass
        elif self.parameters.isotropic:
            transform = torchio.transforms.CropOrPad((220, 320, 320))
            subject = transform(subject)
        else:
            if self.parameters.input_size == 'small':
                transform = torchio.transforms.CropOrPad((130, 250, 350))
            else:
                transform = torchio.transforms.CropOrPad((190, 448, 448))
            subject = transform(subject)
        
        # Augmentations
        if self.phase == "training":
            if self.parameters.aug_policy == 'none':
                pass
            elif self.parameters.aug_policy == 'randaugment':
                randaug_N = 3  # number of augmentations
                randaug_M = 7  # magnitude
                assert randaug_M < 10

                selected_augs = []

                # Magnitude ranges
                randaug_dict = {
                    "affine_translation": np.linspace(0, 10, 10),
                    "blur_std": np.linspace(0, 5, 10),
                    "log_gamma": np.linspace(0, 0.4, 10),
                    "noise": {
                        "mean": np.linspace(0, 0.2, 10),
                        "std": np.linspace(0.15, 0.7, 10)
                    }
                }

                ops = random.sample(list(randaug_dict), k=randaug_N)  # select N random augs
                
                # Add the affine trassformation.
                # affine scaling and rotations are applied with a constant probability
                # but with different magnitudes
                if "affine_translation" in ops:
                    affine_translation = randaug_dict['affine_translation'][randaug_M]
                else:
                    affine_translation = 0
                selected_augs.append(
                    torchio.transforms.RandomAffine(
                        scales=np.linspace(0, 0.3, 10)[randaug_M],  # magnitude range for scaling
                        degrees=np.linspace(0, 30, 10)[randaug_M],  # magnitude range for rotations
                        translation=affine_translation,
                        p=0.65  # constant probability
                    )
                )

                for o in ops:
                    if o == 'affine_translation':
                        pass  # already added
                    elif o == 'blur_std':
                        selected_augs.append(
                            torchio.transforms.RandomBlur(std=randaug_dict['blur_std'][randaug_M])
                        )
                    elif o == 'log_gamma':
                        selected_augs.append(
                            torchio.transforms.RandomGamma(log_gamma=randaug_dict['log_gamma'][randaug_M])
                        )
                    elif o == 'noise':
                        selected_augs.append(
                            torchio.transforms.RandomNoise(
                                mean=randaug_dict['noise']['mean'][randaug_M],
                                std=randaug_dict['noise']['std'][randaug_M]
                            )
                        )
                
                augmenter = torchio.transforms.Compose(selected_augs)
                subject = augmenter(subject)
            else:
                is_horizontal_flip = random.choice([0.0, 1.0])
                if self.parameters.aug_policy == 'affine':
                    # affine + LR flip
                    augmenter = torchio.transforms.Compose([
                        torchio.transforms.RandomFlip(axes=2, flip_probability=is_horizontal_flip),
                        torchio.transforms.RandomAffine(
                            scales=self.parameters.affine_scale,
                            degrees=self.parameters.affine_rotation_deg,
                            translation=self.parameters.affine_translation,
                            p=0.65
                        )
                    ])
                elif self.parameters.aug_policy == 'policy1':
                    augmenter = torchio.transforms.Compose([
                        torchio.transforms.RandomFlip(axes=2, flip_probability=is_horizontal_flip),
                        torchio.transforms.RandomAffine(scales=(0.95,1.05), degrees=5, p=0.50),
                        torchio.OneOf({
                            torchio.transforms.RandomBlur(): 0.5,
                            torchio.transforms.RandomNoise(): 0.5
                        }, p=0.5),
                        torchio.OneOf({
                            torchio.transforms.RandomMotion(): 0.20,
                            torchio.transforms.RandomGhosting(): 0.40,
                            torchio.transforms.RandomSpike(): 0.40
                        }, p=0.5)
                    ])
                elif self.parameters.aug_policy == 'motion':
                    augmenter = torchio.transforms.Compose([
                        torchio.transforms.RandomFlip(axes=2, flip_probability=is_horizontal_flip),
                        torchio.transforms.RandomAffine(scales=(0.90,1.10), p=0.65),
                        torchio.transforms.RandomMotion(p=0.5)
                    ])
                elif self.parameters.aug_policy == 'ghosting':
                    augmenter = torchio.transforms.Compose([
                        torchio.transforms.RandomFlip(axes=2, flip_probability=is_horizontal_flip),
                        torchio.transforms.RandomAffine(scales=(0.90,1.10), p=0.65),
                        torchio.transforms.RandomGhosting(p=0.5)
                    ])
                elif self.parameters.aug_policy == 'spike':
                    augmenter = torchio.transforms.Compose([
                        torchio.transforms.RandomFlip(axes=2, flip_probability=is_horizontal_flip),
                        torchio.transforms.RandomAffine(scales=(0.90,1.10), p=0.65),
                        torchio.transforms.RandomSpike(p=0.5)
                    ])
                else:
                    raise ValueError(f"Unknown type of augmentation policy {self.parameters.aug_policy}")
                if is_horizontal_flip == 1.0 and self.parameters.label_type == 'cancer':
                        # Flip labels to match sides after horizontal flip
                        label_old = label.clone()
                        label[0] = label_old[2] #right benign
                        label[1] = label_old[3] #right malignant
                        label[2] = label_old[0] #left benign
                        label[3] = label_old[1] #left malignant

                subject = augmenter(subject)
                #print("*** After augmentation:", subject.shape)
        
        # Get final volume, perform subtraction if applicable
        if self.parameters.input_type == 'sub_t1c1':
            input_volume = subject['sub_t1c1'].data
        elif self.parameters.input_type == 'sub_t1c2':
            input_volume = subject['sub_t1c2'].data
        elif self.parameters.input_type == 't1c1':
            input_volume = subject["post1"].data
        elif self.parameters.input_type == 't1c2':
            input_volume = subject["post2"].data
        elif self.parameters.input_type == 't1pre':
            input_volume = subject["pre"].data
        elif self.parameters.input_type == 't2':
            input_volume = subject["t2"].data
        elif self.parameters.input_type == 'three_channel':
            if self.parameters.age_as_channel:
                input_volume = torch.stack([
                    subject['pre'].data,
                    subject['post1'].data,
                    subject['post2'].data,
                    torch.empty_like(subject['pre'].data).fill_(datalist['age'])
                ]).squeeze()
            else:
                input_volume = torch.stack([
                    subject['pre'].data,
                    subject['post1'].data,
                    subject['post2'].data
                ]).squeeze()
        elif self.parameters.input_type == 'five_channel':
            subtraction1 = subject['post1'].data - subject['pre'].data
            subtraction2 = subject['post2'].data - subject['pre'].data
            
            input_volume = torch.stack([
                subject['pre'].data,
                subject['post1'].data,
                subject['post2'].data,
                subtraction1,
                subtraction2
            ]).squeeze()
        elif self.parameters.input_type == 'mip_t1c2':
            input_volume = torch.squeeze(subject.data)
            if torch.isnan(input_volume).any():
                raise ValueError("Nan in input volume")
            t = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop(size=(448, 448), pad_if_needed=True),
                torchvision.transforms.ToTensor()
            ])
            input_volume = t(input_volume)
            #print("*** after crop: ", input_volume.shape)
        elif self.parameters.input_type in ['random', 'multi']:
            subtraction1 = (subject['post1'].data - subject['pre'].data)
            subtraction2 = (subject['post2'].data - subject['pre'].data)

            input_volume = torch.stack([
                subject['pre'].data,
                subject['post2'].data,
                subtraction1,
            ]).squeeze().unsqueeze(1)
        
        elif self.parameters.input_type == 'MIL':
            if item_t2 is not None:
                input_volume = torch.stack([
                    subject['sub_t1c1'].data,
                    subject['t2'].data
                ]).squeeze()
            else:
                input_volume = subject['sub_t1c1'].data

        input_volume = input_volume.float()  # Necessary for mean/std calculation

        # Cutout
        if self.parameters.cutout and self.phase == 'training':
            cutout = Cutout3D(
                p=1.0,
                fill='mean',
                cutout_percentage_z=self.parameters.cutout_percentage,
                cutout_percentage_x=self.parameters.cutout_percentage,
                cutout_percentage_y=self.parameters.cutout_percentage
            )
            input_volume = cutout(input_volume)

        # Z-normalize
        mean, std = input_volume.mean(), input_volume.std()
        input_volume -= mean
        input_volume /= std

        # Rescale to 0,1
        input_volume -= input_volume.min()
        input_volume /= input_volume.max()

        if self.parameters.input_type == 'mip_t1c2':
            input_volume = torch.squeeze(input_volume).unsqueeze(0).repeat(3,1,1)
        else:
            input_volume = torch.squeeze(input_volume)#.float()  # from (1,z,x,y) to (z,x,y)
        
        return item_key, input_volume, label

    def __getitem__(self, index):
        item_key, input_volume, label = self.get_datum(index)

        mixed_label = 0  # label for mixup
        if self.parameters.mixup and self.phase == 'training':
            rand_index = randrange(len(self.data_list))
            item_key2, input_volume2, label2 = self.get_datum(rand_index)
            ratio = 0.5
            mixed_volume = (ratio * input_volume) + ((1-ratio) * input_volume2)
            mixed_label = torch.stack([label, label2])

            input_volume = mixed_volume
        
        return item_key, input_volume, label, mixed_label

    def __len__(self):
        return len(self.data_list)

    def get_labels(self):
        number_of_examples = len(self.data_list)

        labels = dict()

        for i in range(0, number_of_examples):
            item_key = self.exam_list[i]
            if self.parameters.label_type == 'birads':
                labels[item_key] = self.birads_dict[item_key]
            elif self.parameters.label_type == 'bpe':
                labels[item_key] = self.bpe_labels_dict[item_key]
            else:
                labels[item_key] = self.data_list[item_key]["label"]

        return labels


def affine_from_sitk(image: sitk.Image):
    """ 
    Generate affine matrix from SimpleITK Image object
    """
    FLIP_XY = np.diag((-1, -1, 1))
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection())
    origin = image.GetOrigin()
    
    if len(direction) == 9:
        rotation = direction.reshape(3, 3)
    elif len(direction) == 4:  # ignore first dimension if 2D (1, W, H, 1)
        rotation_2d = direction.reshape(2, 2)
        rotation = np.eye(3)
        rotation[:2, :2] = rotation_2d
        spacing = *spacing, 1
        origin = *origin, 0
    else:
        raise RuntimeError(f'Direction not understood: {direction}')
    
    rotation = np.dot(FLIP_XY, rotation)
    rotation_zoom = rotation * spacing
    translation = np.dot(FLIP_XY, origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation
    
    return affine


class Cutout3D(object):
    """A class that makes a 3D (cuboid) cutout in the given image
    an extension of the original 2d cutout as described in:
    https://arxiv.org/pdf/1708.04552.pdf
    """

    def __init__(self,
                 p=1.0,
                 cutout_percentage_z = 0.4,
                 cutout_percentage_x = 0.4,
                 cutout_percentage_y = 0.4,
                 fill='mean'):
        """ Construct cutout object

        :param p: Probability that the cutout will be applied
        :param cutout_percentage_z: How much of the input image (percentage-wise)
            can be cut out in the Z axis (first dimension)
            Has to be in a range (0, 1), which represents (0%, 100%)
        :param cutout_percentage_x: How much of the input image (percentage-wise)
            can be cut out in the x axis (second dimension)
        :param cutout_percentage_y: How much of the input image (percentage-wise)
            can be cut out in the y axis (third dimension)
        :param fill: What value will be assigned to the cutout block. Options:
            minimum - fill the block with minimum value of the volume
            zero - fill the block with zeros
            mean - fill the block with the mean value of given volume
            max - fill the block with maximum value of given volume
        """
        self.p = p
        self.cutout_percentage_z = cutout_percentage_z
        self.cutout_percentage_x = cutout_percentage_x
        self.cutout_percentage_y = cutout_percentage_y
        assert fill in ['minimum', 'zero', 'mean', 'max']
        self.fill = fill
        
        
    def __call__(self, volume):
        """ Apply cutout

        :param volume: Volume to apply cutout to
            Expects to have shape of (1, Z, X, Y)
        """
        # Expects volume of size (1,Z,X,Y)
        assert volume.ndim == 4, f"Input expected to be (1,Z,X,Y), instead received {volume.ndim} dimensions"
        
        if random.random() > self.p:
            return volume
        
        # calculate cutout size from percentages
        cutout_size_z = round(volume.shape[1] * self.cutout_percentage_z)
        cutout_size_x = round(volume.shape[2] * self.cutout_percentage_x)
        cutout_size_y = round(volume.shape[3] * self.cutout_percentage_y)
        
        # center point of cutout block
        middle_z = random.randint(0, volume.shape[1])
        middle_x = random.randint(0, volume.shape[2])
        middle_y = random.randint(0, volume.shape[3])
        
        # boundaries of cutout block
        start_z = np.maximum(0, middle_z - cutout_size_z // 2)
        end_z = np.minimum(volume.shape[1], middle_z + cutout_size_z // 2)
        
        start_x = np.maximum(0, middle_x - cutout_size_x // 2)
        end_x = np.minimum(volume.shape[2], middle_x + cutout_size_x // 2)
        
        start_y = np.maximum(0, middle_y - cutout_size_y // 2)
        end_y = np.minimum(volume.shape[3], middle_y + cutout_size_y // 2)
        
        # Fill:
        if self.fill == 'minimum':
            volume[:, start_z:end_z, start_x:end_x, start_y:end_y] = volume.min()
        elif self.fill == 'zero':
            volume[:, start_z:end_z, start_x:end_x, start_y:end_y] = 0
        elif self.fill == 'mean':
            volume[:, start_z:end_z, start_x:end_x, start_y:end_y] = volume.mean()
        elif self.fill == 'max':
            print("Volume max:", volume.max())
            volume[:, start_z:end_z, start_x:end_x, start_y:end_y] = volume.max()
        
        return volume
