import numpy as np

def inside_circle(point, radius=5):
  if point[0]**2+point[1]**2 <= radius**2:
    return True
  return False
  
#Removing own ego vehicle points from the point cloud by cosidering a circular frame
def remove_ego_car(pcd):  
    index = []
    for i,point in enumerate(pcd):
        if inside_circle(point,1.5):
            index.append(i)

    return np.delete(pcd, index, axis=0)

import random

def ransac(pointCloud, distance_threshold = 0.3, P=0.99):
  max_point_num = -999  
  iter = 0
  max_iter = 10
  while iter<max_iter:
    select = random.sample(range(len(pointCloud)),3)
    if abs(pointCloud[select[0],1] - pointCloud[select[1],1]) < 3:
      continue
    coefficiants = predict_plane(pointCloud[select,:])
    if coefficiants is None:
      continue
    r = np.sqrt(coefficiants[0]**2+coefficiants[1]**2+coefficiants[2]**2)
    d = np.divide(np.abs(np.matmul(coefficiants[:3], pointCloud.T[0:3]) + coefficiants[3]) , r)
    d_filt = np.array(d < distance_threshold)
    near_num_point = np.sum(d_filt, axis=0)

    if near_num_point > max_point_num:
      max_point_num = near_num_point
      best_filt = d_filt
      best_model = coefficiants

      w = near_num_point/len(pointCloud)
      wn = np.power(w, 3)
      p_no_outliers = 1.0 - wn
      max_iter = (np.log(1-P) / np.log(p_no_outliers))


    iter+=1
  return np.argwhere(best_filt).flatten(), best_model

def predict_plane(xyz):
  vector1 = xyz[1,:] - xyz[0,:]
  vector2 = xyz[2,:] - xyz[0,:]

  if not np.all(vector1):
      print('will divide by zero..')
      return None
  dy1dy2 = vector2 / vector1
  if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
      return None


  a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
  b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
  c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])

  d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
  return np.array([a,b,c,d])

def remove_ground(pc):
    pc = remove_ego_car(pc)
    indices = ransac(pc)
    return np.delete(pc, indices[0], axis=0)