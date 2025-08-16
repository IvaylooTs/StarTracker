import numpy as np
from scipy.spatial.transform import Rotation as rotate


def star_coords_to_unit_vector(star_coords, center_coords, f_x, f_y):
    """
    Unit vector function -> finds star unit vectros based on star pixel coordinates
    using pinhole projection
    We're only finding the direction where the 3D object lies since we don't know the actual depth.
    Inputs:
    - star_coords: array of tuples that holds (x, y) pixel coordinates of stars in the image
    - center_coords: image center (principal point) coordinates in tuple form: (xc, yc)
    - f_x: const. Describes the scaling factor by x
    - f_y: const. Describes the scaling factor by y
    Outputs:
    - unit_vectors: array of np arrays holding (x, y, z) coordinates of each unit vector
    """
    # Unpack center_coords
    c_x, c_y = center_coords

    star_coords_matrix = np.array(star_coords)
    # Initialize empty unit vector array
    unit_vectors = []

    # Reverse pinhole projection to get X, Y, Z coords of each star

    # Convert pixel coordinates into normalized camera coordinates (back-projection step)
    # These correspond to where the pixel projects onto the image plane at depth z_p
    # Inversing pinhole projection formula for x (back-projecting). Formula: x = f_x * (x_p / z_p) + c_x
    x_norm = (star_coords_matrix[:, 0] - c_x) / f_x
    # Inversing pinhole projection formula for y (back-projecting). Formula: y = f_y * (x_p / z_p) + c_y
    y_norm = (star_coords_matrix[:, 1] - c_y) / f_y
    # Pick arbitrary depth for each 3D point
    # (we don't know how far away they really are, we just want their direction)
    z_coord = np.ones_like(x_norm)

    # array holding [X, Y, Z] coordinates of the direction vector
    vectors = np.stack([x_norm, y_norm, z_coord], axis=1)

    # Normalize vector to save on calculations later on
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms

    return unit_vectors


def image_vector_matrix(image_unit_vectors):
    return np.array(image_unit_vectors, dtype=np.float64)


def catalog_vector_matrix(final_solution, catalog_unit_vectors):
    matrix = []
    for _, hip in final_solution[0].items():
        cat_vector = catalog_unit_vectors.get(hip)
        if cat_vector is None:
            raise ValueError(f"Catalog vector for HIP {hip} not found.")
        matrix.append(cat_vector)
    return np.array(matrix, dtype=np.float64)


def inverse_rotate_vectors(quaternion, catalog_vectors):
    """
    Function to apply inverse rotation of quaternion to our catalog vectors. We should
    get vectors in our camera frame
    Inputs:
    - quaternion: rotation in [w, x, y, z] format
    - catalog_vectors: the unit vectors of the assigned HIP IDs stars
    Output:
    - camera_frame_vectors: vectors in camera frame
    """
    # Convert from [w, x, y, z] to [x, y, z, w] for scipy
    q_scalar_last = np.roll(quaternion, -1)
    rot_object = rotate.from_quat(q_scalar_last)
    camera_frame_vectors = rot_object.inv().apply(catalog_vectors)
    return camera_frame_vectors


def project_to_image_plane(vectors, params):
    """
    Function to project vectors in our camera frame to pixel coordinates in the
    image plane
    Inputs:
    - vectors: array of vectors to project
    - params: array of camera params:
      - params[0] = (f_x, f_y)
      - params[1] = (c_x, c_y)
    Outputs:
    - projected_coords: coordinates of each point. If point is behind camera
    the element in the array is None
    """
    f_x, f_y = params[0]
    c_x, c_y = params[1]
    projected_coords = []
    for vector in vectors:
        x, y, z = vector
        if z <= 0:
            projected_coords.append(None)
        else:
            projected_x = f_x * (x / z) + c_x
            projected_y = f_y * (y / z) + c_y
            projected_coords.append((projected_x, projected_y))

    return projected_coords


def reproject_vectors(quaternion, catalog_vectors, params):
    """
    Function that reprojects the assigned catalog unit vectors back into
    our image plane
    """
    camera_vectors = inverse_rotate_vectors(quaternion, catalog_vectors)
    projected_coords = project_to_image_plane(camera_vectors, params)
    return projected_coords

