import cv2
import numpy as np
import torch 

from lib.zenlib.utils.torch_geometry_transforms import *


def smpl_mat_to_aa(poses):
    poses_aa = []
    for pose_frame in poses:
        pose_frames = []
        for joint in pose_frame:
            pose_frames.append(cv2.Rodrigues(joint)[0].flatten())
        pose_frames = np.array(pose_frames)
        poses_aa.append(pose_frames)
    poses_aa = np.array(poses_aa)
    return poses_aa


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3

    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    zeros = torch.zeros(z.shape, dtype = z.dtype).to(ortho6d.device)
    matrix = torch.cat((x,y,z, zeros), 2) #batch*3*3
    return matrix

def compute_orth6d_from_rotation_matrix(rot_mats):
    if len(rot_mats.shape) == 3:
        rot_mats = rot_mats[:,:,:2].transpose(1, 2).reshape(-1, 6)
    else:
        rot_mats = rot_mats[:,:2].transpose(0, 1).reshape(-1, 6)
    return rot_mats
    
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_quat_to_orth6d(poses):
    rot_mats = compute_rotation_matrix_from_quaternion(poses.to(poses.device).float().view(-1, 4))
    rot_mats = rot_mats[:,:3,:]
    orth6d = compute_orth6d_from_rotation_matrix(rot_mats)
    orth6d = orth6d.view(poses.shape[0], -1, 6) 
    return orth6d

def compute_orth_6d_to_quat(orth6d):
    orth6d_flat = orth6d.view(-1, 6)
    rot_mat6d = compute_rotation_matrix_from_ortho6d(orth6d_flat)
    pose_aa = rotation_matrix_to_quaternion(rot_mat6d)
    pose_aa = pose_aa.reshape(orth6d.shape[0], -1) 
    return pose_aa

def compute_aa_to_orth6d(poses):
    rot_mats = angle_axis_to_rotation_matrix(torch.tensor(poses).to(poses.device).float().view(-1, 3))
    rot_mats = rot_mats[:,:3,:]
    orth6d = compute_orth6d_from_rotation_matrix(rot_mats)
    orth6d = orth6d.view(poses.shape[0], -1, 6) 
    return orth6d

def compute_orth_6d_to_aa(orth6d):
    orth6d_flat = orth6d.view(-1, 6)
    rot_mat6d = compute_rotation_matrix_from_ortho6d(orth6d_flat)
    pose_aa = rotation_matrix_to_angle_axis(rot_mat6d)
    pose_aa = pose_aa.reshape(orth6d.shape[0], -1) 
    return pose_aa

def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.tensor([1e-8], dtype = v_mag.dtype).to(v.device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

#quaternion batch*4
def compute_rotation_matrix_from_quaternion(quaternion):
    batch=quaternion.shape[0]
    
    
    quat = normalize_vector(quaternion).contiguous()
    
    qw = quat[...,0].contiguous().view(batch, 1)
    qx = quat[...,1].contiguous().view(batch, 1)
    qy = quat[...,2].contiguous().view(batch, 1)
    qz = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    
#axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle( axisAngle):
    batch = axisAngle.shape[0]
    
    theta = torch.tanh(axisAngle[:,0])*np.pi #[-180, 180]
    sin = torch.sin(theta)
    axis = normalize_vector(axisAngle[:,1:4]) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 (x,y,z)*theta
def compute_rotation_matrix_from_Rodriguez( rod):
    batch = rod.shape[0]
    
    axis, theta = normalize_vector(rod, return_mag=True)
    
    sin = torch.sin(theta)
    
    
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf( hopf):
    batch = hopf.shape[0]
    
    theta = (torch.tanh(hopf[:,0])+1.0)*np.pi/2.0 #[0, pi]
    phi   = (torch.tanh(hopf[:,1])+1.0)*np.pi     #[0,2pi)
    tao   = (torch.tanh(hopf[:,2])+1.0)*np.pi     #[0,2pi)
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    

#euler batch*4
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix


#euler_sin_cos batch*6
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch=euler_sin_cos.shape[0]
    
    s1 = euler_sin_cos[:,0].view(batch,1)
    c1 = euler_sin_cos[:,1].view(batch,1)
    s2 = euler_sin_cos[:,2].view(batch,1)
    c2 = euler_sin_cos[:,3].view(batch,1)
    s3 = euler_sin_cos[:,4].view(batch,1)
    c3 = euler_sin_cos[:,5].view(batch,1)

        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(GLOBAL_DEVICE)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(GLOBAL_DEVICE))*-1 )
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):
    
    batch=m.shape[0]
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(GLOBAL_DEVICE)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(GLOBAL_DEVICE))*-1 )
    
    theta = torch.acos(cos)
    
    return theta
    
def get_sampled_rotation_matrices_by_quat(batch):
    #quat = torch.autograd.Variable(torch.rand(batch,4).to(GLOBAL_DEVICE))
    quat = torch.autograd.Variable(torch.randn(batch, 4).to(GLOBAL_DEVICE))
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix
    
def get_sampled_rotation_matrices_by_hpof(batch):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,1, batch)*np.pi).to(GLOBAL_DEVICE)) #[0, pi]
    phi   =  torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).to(GLOBAL_DEVICE))      #[0,2pi)
    tao   = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).to(GLOBAL_DEVICE))      #[0,2pi)
    
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*4 angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle( batch, return_quaternion=False):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(-1,1, batch)*np.pi).to(GLOBAL_DEVICE)) #[0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).to(GLOBAL_DEVICE))
    axis = normalize_vector(axis) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    quaternion = torch.cat((qw.view(batch,1), qx.view(batch,1), qy.view(batch,1), qz.view(batch,1)), 1 )
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    if(return_quaternion==True):
        return matrix, quaternion
    else:
        return matrix


    

    
    
    
    
    