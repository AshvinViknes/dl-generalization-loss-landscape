# utils.py
import logging
import colorlog
import json
from datetime import datetime
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

log = logging.getLogger("utils")

def setup_logger():
    """
    Configure logging for the application with both console and file output.
    Sets up a dual-handler logging system:
    - Console handler: Outputs colored, formatted logs to stdout with timestamps
    - File handler: Outputs non-colored logs to 'logs/run.log' with timestamps
    Creates a 'logs' directory if it doesn't exist. Log level is set to INFO.
    Console output uses color coding: INFO (cyan), WARNING (yellow), ERROR (red), DEBUG (white).
    Returns:
        None
    Raises:
        OSError: If the 'logs' directory cannot be created.
    """

    os.makedirs("logs", exist_ok=True)
    # console handler with colored output and timestamp
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={"INFO":"cyan","WARNING":"yellow","ERROR":"red","DEBUG":"white"}
    ))
    # file handler with timestamped, non-colored logs
    file_handler = logging.FileHandler("logs/run.log")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[handler, file_handler])



def log_json(event, **data):
    """
    Log an event with associated data to a JSON Lines file.
    
    Creates a 'logs' directory if it doesn't exist and appends a JSON object
    containing a timestamp, event name, and additional data to 'logs/events.jsonl'.
    
    Args:
        event (str): The name or description of the event to log.
        **data: Arbitrary keyword arguments containing additional data to include
                in the log entry.
    
    Returns:
        None
    
    Example:
        >>> log_json("user_login", user_id=123, ip_address="192.168.1.1")
        # Appends to logs/events.jsonl:
        # {"timestamp": "2026-01-26T20:15:39.809436", "event": "epoch", "epoch": 1, "loss": 1.636197941271464, "acc": 0.9467}
    """

    os.makedirs("logs", exist_ok=True)
    payload={"timestamp":datetime.utcnow().isoformat(),"event":event,**data}
    with open("logs/events.jsonl","a") as f: f.write(json.dumps(payload)+"\n")

def load_mnist():
    """
    Load the MNIST dataset and prepare data loaders.
    This function downloads and loads the MNIST dataset, splitting the training
    set into two halves: a clean half for evaluation and a remaining half for
    potential contamination or further processing.
    Returns
    -------
    tuple
        A tuple containing:
        - train (Dataset): The full MNIST training dataset with tensor transformations.
        - test (Dataset): The MNIST test dataset with tensor transformations.
        - eval_clean (DataLoader): DataLoader for the clean half of training data
          with batch size 256 and shuffle=False.
        - list: Indices of the second half of the training set (for potential
          contamination or alternative use).
    Notes
    -----
    - Logs the total number of training samples and clean samples.
    - Converts images to tensors using torchvision transforms.
    - Downloads datasets to "./data" directory if not already present.
    """

    tfm = transforms.ToTensor()
    log.info("Loading MNIST…")
    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    n=len(train); half=n//2
    clean_half=Subset(train, list(range(half)))
    eval_clean=DataLoader(clean_half, batch_size=256, shuffle=False)
    log.info(f"Train={n}, Clean={half}")
    return train, test, eval_clean, list(range(half,n))

loss_fn=torch.nn.CrossEntropyLoss()

@torch.inference_mode()
def eval_loss(model, loader, device):
    """
    Calculate the average loss of a model on a dataset.
    Args:
        model: The neural network model to evaluate.
        loader: DataLoader containing the dataset (batches of x, y pairs).
        device: The device (CPU or GPU) on which to perform computation.
    Returns:
        float: The average loss across all samples in the loader.
    """

    total=0; n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        total+=loss_fn(model(x),y).item()*x.size(0); n+=x.size(0)
    return total/n

@torch.inference_mode()
def eval_acc(model, loader, device):
    """
    Evaluate the accuracy of a model on a given dataset.
    Args:
        model: The neural network model to evaluate.
        loader: A DataLoader containing batches of (input, target) pairs.
        device: The device (CPU or GPU) to run the model on.
    Returns:
        float: The accuracy as a fraction between 0 and 1.
    """

    correct=0; n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        correct+=(model(x).argmax(1)==y).sum().item(); n+=x.size(0)
    return correct/n

def apply_corruption(train_set, original_targets, corrupt_idx, p, seed=0):
    """
    Apply label corruption to a training dataset by randomly flipping labels.
    Args:
        train_set: Dataset object with a 'targets' attribute to be corrupted.
        original_targets: Tensor of original uncorrupted target labels.
        corrupt_idx: List or array of indices indicating which samples are candidates for corruption.
        p: Probability (float between 0 and 1) that each candidate sample will be corrupted.
        seed: Random seed for reproducibility (default: 0).
    Returns:
        int: Number of samples actually corrupted.
    Notes:
        - Corrupted labels are replaced with random integers in the range [0, 10).
        - Original targets are restored before applying new corruption.
        - Logs corruption probability and total number of corrupted samples.
    """

    log.info(f"Applying corruption p={p}")
    train_set.targets=original_targets.clone()
    if p<=0: return 0
    rng=np.random.default_rng(seed)
    targets=train_set.targets.numpy()
    mask=rng.random(len(corrupt_idx))<p
    idxs=np.array(corrupt_idx)[mask]
    for i in idxs: targets[i]=rng.integers(0,10)
    train_set.targets=torch.from_numpy(targets).long()
    log.info(f"Corrupted={len(idxs)}")
    return len(idxs)

def train_model(model, train_loader, test_loader, epochs, lr, device):
    """
    Trains a neural network model on the provided training data and evaluates it on test data.
    This function trains the model using the Adam optimizer with an initial learning rate that
    is halved at epoch 7. The training loss and test accuracy are recorded for each epoch.
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        epochs (int): Number of training epochs.
        lr (float): Initial learning rate for the Adam optimizer.
        device: The device (CPU or GPU) to run the training on.
    Returns:
        tuple: A tuple containing:
            - train_losses (list): Average training loss for each epoch.
            - test_accs (list): Test accuracy for each epoch.
    """

    opt=torch.optim.Adam(model.parameters(), lr=lr)
    train_losses=[]; test_accs=[]
    for ep in range(1,epochs+1):
        if ep==7:
            for g in opt.param_groups: g["lr"]*=0.5
            log.info("LR halved")
        model.train()
        total=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            out=model(x)
            loss=loss_fn(out,y)
            loss.backward()
            opt.step()
            total+=loss.item()*x.size(0)
        avg=total/len(train_loader.dataset)
        acc=eval_acc(model,test_loader,device)
        train_losses.append(avg); test_accs.append(acc)
        log.info(f"Epoch {ep}: loss={avg:.4f}, acc={acc*100:.2f}")
        log_json("epoch",epoch=ep,loss=avg,acc=float(acc))
    return train_losses, test_accs

def save_csv(rows):
    """
    Save a list of rows to a CSV file in the results directory.
    Creates a 'results' directory if it doesn't exist, then writes the provided
    rows to a CSV file named 'summary.csv' with the following column headers:
    'p', 'clean_loss', 'test_acc', 'r_max', 'area'.
    Args:
        rows (list): A list of tuples or lists where each element represents
                     a row of data to be written to the CSV file. Each row
                     should contain values corresponding to the columns:
                     p, clean_loss, test_acc, r_max, and area.
    Returns:
        None
    """

    os.makedirs("results",exist_ok=True)
    import csv
    with open("results/summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["p","clean_loss","test_acc","r_max","area"]); w.writerows(rows)
