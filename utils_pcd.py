import os
import numpy as np
from pypcd import pypcd as pcd

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def roll_pitch_yaw(ax, ay, az):
  cx, sx = np.cos(ax), np.sin(ax)
  cy, sy = np.cos(ay), np.sin(ay)
  cz, sz = np.cos(az), np.sin(az)
  
  R = np.array([[cz*cy, cz*sx*sy-sz*cx, cz*sx*sy+sz*sy],
                [sz*cy, sz*sx*sy+cz*cx, sz*sx*sy-cz*sx],
                [-sy, cy*sx, cy*cx]])
  return R


def add_noise(pc, mean, std):
  return pc + np.random.normal(loc=mean, scale=std, size=pc.shape)


def save_whole_cloud(paths_in, path_out):
  start = True
  whole = None
  for path_in in paths_in:
    pc = pcd.point_cloud_from_path(path_in)
    meta = pc.get_metadata()
    trans = np.array(meta['viewpoint'][:3])
    rot = quaternion_rotation_matrix(meta['viewpoint'][3:]).T
    
    cloud = np.vstack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']])
    
    rotated = rot.T @ cloud + trans.reshape(-1,1)
    
    pc.pc_data['x'] = rotated[0,:]
    pc.pc_data['y'] = rotated[1,:]
    pc.pc_data['z'] = rotated[2,:]
    if start:
      start = False
      whole = pc
    else:
      whole = pcd.cat_point_clouds(whole, pc)
      
  whole.save_pcd(path_out)
  
  
    
