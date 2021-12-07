import numpy as np
import torch

def mixup(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# memory, saving, loading
import gc
def clear_mem(model, model_name='model'):
    """
    Clears GPU cache in notebook
    """
    if model_name in locals():
        print('deleting model...')
        del model
    for x in list(globals().keys()):
        variable = eval(x)
        if torch.is_tensor(variable) and variable.is_cuda:
            print(x)
            del variable
    gc.collect()
    torch.cuda.empty_cache()

# save model file
def save_model(model, save_path, **metrics):
    """
    Save the model to the path directory provided
    """
    if "state_dict" in metrics:
        raise Warning("We will use states from the model instead.")
        del metrics["state_dict"]
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'state_dict': model_to_save.state_dict()}
    checkpoint.update(metrics)
    torch.save(checkpoint, save_path)
    return save_path, metrics

# load model file
def load_model(save_path, model_class=None, model=None):
    """
    Load the model from the path directory provided
    """
    if model is None:
        if model_class is None:
            raise ValueError("No model to construct!")
        model = model_class()
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
    metrics = {k:checkpoint[k] for k in checkpoint if k!='state_dict'}

    return model, metrics