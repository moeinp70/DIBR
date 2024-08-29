import numpy as np
import cv2
from dibr import DIBR


N = 5


def generate_multiple_views(V_o, D_o, K_o, Rt_o, K_v, Rt_v, Znear, Zfar, N):
    """
    Returns:
        list: List of synthesized virtual views.
    """
    views = []
    for i in range(N):
        t = i / (N - 1)
        Rt_interpolated = (1 - t) * Rt_o + t * Rt_v
        view = DIBR(V_o, D_o, K_o, Rt_o, K_v, Rt_interpolated, Znear, Zfar)
        views.append(view)
        print(f"View {i+1} Done")

    return views



# Example usage:
image_path = 'V_original.png'
depth_map_path = 'D_original.png'
V_original = cv2.imread(image_path)
D_original = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

# Camera parameters (Intrinsic and Extrinsic matrices)
K_original = np.array([[1732.87, 0.0, 943.23], 
                       [0.0, 1729.90, 548.845040], 
                       [0, 0, 1]])

Rt_original = np.array([[1.0, 0.0, 0.0, 0], 
                        [0.0, 1.0, 0.0, 0.0], 
                        [0.0, 0.0, 1.0, 0.0]])

K_virtual = np.array([[1732.87, 0.0, 943.23], 
                      [0.0, 1729.90, 548.845040], 
                      [0, 0, 1]])

Rt_virtual = np.array([[1.0, 0.0, 0.0, 1.5924], 
                       [0.0, 1.0, 0.0, 0.0], 
                       [0.0, 0.0, 1.0, 0.0]])

Zfar = 2760.510889
Znear = 34.506386


views = generate_multiple_views(V_original, D_original, K_original, Rt_original, K_virtual, Rt_virtual, Znear, Zfar, N)

# Save each view
for idx, view in enumerate(views):
    output_image_path = f'view_{idx}.png'
    cv2.imwrite(output_image_path, view)
    print(f"View {idx} saved to {output_image_path}")
