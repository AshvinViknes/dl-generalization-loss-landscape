# landscape.py - Functions for computing loss landscape curves along random directions in parameter space.

import torch
import logging
import numpy as np

log=logging.getLogger("landscape")

def make_direction(model, seed=0):
    """
    Generate a random direction vector for model parameters with normalized gradient.
    This function creates a random direction in the parameter space of a neural network model.
    The direction is normalized by the parameter norms to account for different weight magnitudes,
    and then normalized globally by the total gradient norm.
    Args:
        model: A PyTorch model whose parameters will be used to generate the direction.
        seed (int, optional): Random seed for reproducibility. Default is 0.
    Returns:
        list: A list of normalized direction tensors, one for each parameter in the model.
              Each tensor has the same shape as the corresponding model parameter.
    Notes:
        - For 2D parameters (e.g., weight matrices), normalization is applied per row.
        - For other dimensional parameters (e.g., biases), normalization is applied globally.
        - The final direction vectors are normalized by the overall gradient norm.
        - A small epsilon (1e-12) is added to prevent division by zero.
    """

    torch.manual_seed(seed)
    vec=[]
    for p in model.parameters():
        v=torch.randn_like(p)
        if v.ndim==2:
            v=v*(p.norm(dim=1,keepdim=True)+1e-12)/(v.norm(dim=1,keepdim=True)+1e-12)
        else:
            v=v*(p.norm()+1e-12)/(v.norm()+1e-12)
        vec.append(v)
    gnorm=torch.sqrt(sum((v*v).sum() for v in vec))
    return [v/(gnorm+1e-12) for v in vec]

def get_base_params(model):
    """
    Extract and return a deep copy of all parameters from the given model.
    This function iterates through all parameters of a PyTorch model,
    detaches them from the computational graph, and clones them to create
    independent copies that are not linked to the original model's parameters.
    Args:
        model: A PyTorch model (nn.Module) whose parameters are to be extracted.
    Returns:
        list: A list of detached and cloned parameter tensors from the model.
              Each parameter is independent and won't be affected by gradient updates
              to the original model.
    """

    return [p.detach().clone() for p in model.parameters()]

@torch.no_grad()
def set_params(model, base, direction, alpha, scale):
    """
    Update model parameters by moving them in a specified direction.
    This function modifies the parameters of a model by adding a scaled directional
    vector to a base parameter set. It's commonly used in optimization algorithms
    like linear interpolation or loss landscape exploration.
    Args:
        model: A PyTorch model whose parameters will be updated.
        base: Iterable of base parameter values (typically model parameters).
        direction: Iterable of directional vectors indicating the change direction
                   for each parameter.
        alpha: Scaling factor for the directional step (controls step magnitude).
        scale: Additional scaling factor applied to the direction vector.
    Returns:
        None. Modifies model parameters in-place.
    """

    for p,p0,d in zip(model.parameters(),base,direction):
        p.copy_(p0+scale*alpha*d)

@torch.no_grad()
def restore(model, base):
    """
    Restore model parameters to a previous state.
    Copies the values from a base set of parameters to the current model parameters.
    This is useful for reverting model weights to a previously saved checkpoint.
    Args:
        model: A PyTorch model whose parameters will be restored.
        base: An iterable of parameter tensors containing the values to restore.
              Should have the same structure and number of parameters as model.parameters().
    Returns:
        None
    """
    
    for p,p0 in zip(model.parameters(),base):
        p.copy_(p0)

def compute_1d_curve(model, base, direction, eval_loss, loader, scale):
    """
    Compute a 1D loss landscape curve along a specified direction.
    This function evaluates the model's loss along a 1D direction in parameter space,
    automatically adapting the search range to find the minimum, then creates a 
    centered plot around that minimum.
    Args:
        model: The neural network model to evaluate.
        base: Base parameters of the model.
        direction: Direction vector in parameter space to traverse.
        eval_loss: Function that evaluates loss given (model, loader).
        loader: Data loader for computing loss values.
        scale: Scaling factor for the direction vector.
    Returns:
        tuple: A tuple containing:
            - centered (np.ndarray): Array of alpha values centered around the minimum.
            - c_losses (np.ndarray): Loss values corresponding to centered alpha values.
            - alpha_star (float): The alpha value at which the minimum loss was found.
    Notes:
        - The function first performs an adaptive search to locate the minimum loss.
        - If the minimum is found at boundary points, the search range is doubled.
        - After finding the minimum, a symmetric range around it is created for detailed analysis.
        - Model parameters are restored to base values after each evaluation.
    """

    R=1.0; num=200
    for _ in range(4):
        alphas=np.linspace(-R,R,num)
        losses=[]
        for a in alphas:
            set_params(model,base,direction,a,scale)
            losses.append(eval_loss(model,loader))
        restore(model,base)
        losses=np.array(losses); imin=int(losses.argmin())
        if imin==0 or imin==num-1: R*=2
        else: break
    alpha_star=float(alphas[imin])
    L=alpha_star-(-R); R2=R-alpha_star; R_sym=max(min(L,R2),1e-6)
    centered=np.linspace(-R_sym,R_sym,num)
    c_losses=[]
    for da in centered:
        a=alpha_star+da
        set_params(model,base,direction,a,scale)
        c_losses.append(eval_loss(model,loader))
    restore(model,base)
    return centered, np.array(c_losses), alpha_star
