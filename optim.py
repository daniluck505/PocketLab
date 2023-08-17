import torch

def make_optimizer(options, model_params):
    params = options['optimizer']
    if params['name'] == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=params['lr'], betas=(params['beta1'], params['beta2']))
    else:
        raise NotImplementedError(f'optimizer {params["optimizer"]} is not implemented')
    return optimizer
