import numpy as np
import torch

from pyquaternion import Quaternion

def compute_geodesic_distance_from_two_matrices(m1, m2, device):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(device))*-1 )
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    return theta

def geodesic_loss(y_hat, y_target, device):
    B = y_hat.shape[0]
    M = y_hat.shape[1]
    loss = torch.zeros((B, M))
    loss.to(device)

    for i in np.arange(0, B):
        m1 = y_hat[i, :, :, :].to(device)
        m2 = y_target[i, :, :, :].to(device)
        loss[i, :] = compute_geodesic_distance_from_two_matrices(m1,  m2, device)
    
    return loss.to(device)

def matrix_transpose_loss(y_hat, y_target, device):
    B = y_hat.shape[0]
    M = y_hat.shape[1]
    loss = torch.zeros((B, M))

    y_target = torch.transpose(y_target, 2, 3)
    loss = torch.norm(torch.matmul(y_hat, y_target), p=2, dim=(2,3))
    return loss.to(device)

def quaternion_loss(y_hat, y_target, device):
    B = y_hat.shape[0]
    M = y_hat.shape[1]
    loss = torch.zeros((B, M))
    loss.to(device)

    for i in np.arange(0, B):
        m1 = y_hat[i, :, :, :].to(device)
        m2 = y_target[i, :, :, :].to(device)
        loss[i, :] = compute_geodesic_distance_from_two_matrices(m1,  m2, device)
    
    return loss.to(device)

def l2_loss(y_hat, y_target):
    loss = torch.norm(y_hat - y_target, dim=1) ** 2
    loss = torch.sum(loss)/loss.shape[0]
    return loss

def matrix_loss(y_hat, y_target, options):
    """
    Arguments:
    ----------
        y_hat: Tensor with size BxMx12 with the M primitives and their corresponding transforms
        y_target: Tensor with size BxMx12 with the M primitives and their corresponding transforms
        options: A dictionary with various options

    Returns:
    --------
        the loss
    """
    
    device = y_hat.device

    B = y_target.shape[0]  # batch size
    M = options["M"]
    rotations_target = y_target[:, :, 3:].view(B, M, 4)
    translations_target = y_target[:, :, :3].view(B, M, 3)

    rotations = y_hat[:, :, 3:].view(B, M, 4)
    translations = y_hat[:, :, :3].view(B, M, 3)
    

    # Translation: L2 norm between translations
    t_loss = l2_loss(translations, translations_target, device)

     # Rotation: L2 distance
    r_loss = l2_loss(rotations, rotations_target, device) #matrix_transpose_loss(rotations, rotations_target, device)

    #print("rot/trans: ", torch.sum(r_loss, dim=1).mean() / (torch.sum(t_loss, dim=1).mean() * 4))

    loss = ((r_loss / 4) + t_loss)

    extra = dict()
    extra["r_loss"] = torch.sum(r_loss, dim=1).mean().detach().cpu().numpy()
    extra["t_loss"] = torch.sum(t_loss, dim=1).mean().detach().cpu().numpy()

    loss = torch.sum(loss, dim=1).mean()
    return loss, extra

