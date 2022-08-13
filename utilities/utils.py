import torch
#from apex import amp


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError(f'{s} is not a valid boolean string')
    return s == 'True'


def save_checkpoint(model, save_path, optimizer=None, is_amp=False, step: int = None, epoch: int = None):
    """
    Save weights
    :param model: Model object
    :param save_path: Where the weights should be saved
    :param is_amp: Half-precision (True if FP16; False if FP32)
    :param step: Provide step number if you want to include it
                 in the filename
    """
    
    # Prepare checkpoints
    # if is_amp:
    #     checkpoint = {
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'amp': amp.state_dict(),
    #         'step': step,
    #         'epoch': epoch
    #     }
    #     suffix = "amp.pt"
    # else:
    #     checkpoint = model.state_dict()
    #     suffix = "pt"
    #checkpoint = model.state_dict()
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'epoch': epoch
    }
    suffix = "pt"
    
    # Create filename
    if step is not None:
        ckp_filename = f"{save_path}-step{step}.{suffix}"
    else:
        ckp_filename = f"{save_path}.{suffix}"
    
    # Save checkpoints
    torch.save(checkpoint, ckp_filename)
