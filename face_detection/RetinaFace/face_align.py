
import cv2
import numpy as np


def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(112, 112)):
    
    reference_pts = get_reference_facial_points(default_square=True)
    ref_pts = np.float32(reference_pts)
    src_pts = np.float32(facial_pts)
    trans_m = get_similarity_transform_for_cv2(src_pts, ref_pts)
    warpped = cv2.warpAffine(src_img, trans_m, (crop_size[0], crop_size[1]))
    
    return warpped


def get_reference_facial_points(default_square=True):
    tmp_size = np.array([96, 112])
    tmp_5pts = np.array([[30.29459953, 51.69630051],
                         [65.53179932, 51.50139999],
                         [48.02519989, 71.73660278],
                         [33.54930115, 92.36550140],
                         [62.72990036, 92.20410156]])
    if default_square:
        size_diff = max(tmp_size) - tmp_size
        tmp_5pts += size_diff / 2

    return tmp_5pts

def get_similarity_transform_for_cv2(src_pts, dst_pts):
    trans, trans_inv = findSimilarity(src_pts, dst_pts)
    #Convert Transform Matrix 'trans' into 'cv2_trans' which could be directly used by cv2.warpAffine()
    cv2_trans = trans[:, 0:2].T
    return cv2_trans


def findSimilarity(uv, xy, options=None):
    options = {'K': 2}
    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)
    # Solve for trans2
    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)
    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    trans2 = np.dot(trans2r, TreflectY)
    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = np.linalg.norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = np.linalg.norm(xy2 - xy)
    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = np.linalg.inv(trans2)
        return trans2, trans2_inv

    
def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {'K': 2}
    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # We know that X * r = U
    if np.linalg.matrix_rank(X) >= 2 * K:
        r, _, _, _ = np.linalg.lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]
    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])
    T = np.linalg.inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return T, Tinv


def tformfwd(trans, uv):
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy

