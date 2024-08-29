import numpy as np
import cv2
from scipy.interpolate import griddata

def DIBR(V_o, D_o, K_o, Rt_o, K_v, Rt_v, Znear, Zfar):
    """
    Function to generate a virtual view using Depth Image-Based Rendering (DIBR).

    Parameters:
        V_o (numpy.ndarray): Original image.
        D_o (numpy.ndarray): Depth map of the original image.
        K_o (numpy.ndarray): Intrinsic matrix of the original camera.
        Rt_o (numpy.ndarray): Extrinsic matrix of the original camera.
        K_v (numpy.ndarray): Intrinsic matrix of the virtual camera.
        Rt_v (numpy.ndarray): Extrinsic matrix of the virtual camera.
        Znear (float): Near clipping plane distance.
        Zfar (float): Far clipping plane distance.

    Returns:
        numpy.ndarray: Synthesized virtual view.
    """

    # Convert depth map to real-world coordinates
    def depth_to_world(D, Znear, Zfar):
        Z = Znear + D * (Zfar - Znear) / 255.0  # Convert to real-world depth
        Z = np.clip(Z, Znear, Zfar)  # Clip depth to avoid extreme values
        return Z

    # Compute the 3D coordinates in the world space
    def image_to_world_coordinates(D, K, Z):
        h, w = D.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))  # Pixel coordinates
        x = (i - K[0, 2]) * Z / K[0, 0]
        y = (j - K[1, 2]) * Z / K[1, 1]
        world_coords = np.stack((x, y, Z), axis=-1)
        return world_coords

    # Project the 3D points onto the virtual camera plane
    def world_to_image_coordinates(world_coords, Rt, K):
        num_points = world_coords.shape[0] * world_coords.shape[1]
        world_coords_homogeneous = np.hstack((world_coords.reshape(-1, 3), np.ones((num_points, 1))))
        camera_coords = world_coords_homogeneous @ Rt.T
        
        # Handle potential outliers by clipping small values to avoid division by near-zero
        camera_coords[camera_coords[:, 2] < 1e-5, 2] = 1e-5
        
        image_coords = camera_coords @ K.T
        image_coords = image_coords[:, :2] / image_coords[:, 2:3]
        return image_coords.reshape(world_coords.shape[0], world_coords.shape[1], 2)

    # Optimized pixel mapping with mixed interpolation and inpainting
    def render_virtual_view_optimized(V_original, image_coords):
        h, w, _ = V_original.shape
        virtual_image = np.zeros_like(V_original)

        # Initial nearest-neighbor mapping to quickly fill most areas
        for y in range(h):
            for x in range(w):
                target_x, target_y = image_coords[y, x]
                if np.isnan(target_x) or np.isnan(target_y):  # Check for NaN values
                    continue
                target_x = np.clip(target_x, 0, w - 1)  # Clip to image bounds
                target_y = np.clip(target_y, 0, h - 1)  # Clip to image bounds
                virtual_image[int(target_y), int(target_x)] = V_original[y, x]
        
        # Identify missing pixels (black regions)
        gray_virtual = cv2.cvtColor(virtual_image, cv2.COLOR_BGR2GRAY)
        missing_pixels = np.where(gray_virtual == 0)

        # Create a sparse grid for missing regions interpolation
        coords_flat = image_coords.reshape(-1, 2)
        V_flat = V_original.reshape(-1, 3)

        # Use only valid points for interpolation
        valid_mask = (coords_flat[:, 0] >= 0) & (coords_flat[:, 0] < w) & (coords_flat[:, 1] >= 0) & (coords_flat[:, 1] < h)
        coords_sampled = coords_flat[valid_mask]
        V_sampled = V_flat[valid_mask]

        # Interpolate missing regions using a mixed strategy
        for channel in range(3):  # For each color channel
            # Use 'linear' interpolation for moderately filled areas
            channel_values = griddata(coords_sampled, V_sampled[:, channel], (missing_pixels[1], missing_pixels[0]), method='linear', fill_value=0)
            virtual_image[missing_pixels[0], missing_pixels[1], channel] = channel_values

            # For small gaps or remaining artifacts, use 'cubic' interpolation
            remaining_mask = virtual_image[..., channel] == 0
            remaining_y, remaining_x = np.where(remaining_mask)
            if len(remaining_y) > 0:
                refined_values = griddata(coords_sampled, V_sampled[:, channel], (remaining_x, remaining_y), method='cubic', fill_value=0)
                virtual_image[remaining_y, remaining_x, channel] = refined_values

        # Inpaint remaining regions to fill small gaps and correct black areas
        mask = (gray_virtual == 0).astype(np.uint8) * 255
        virtual_image = cv2.inpaint(virtual_image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        return virtual_image

    # Run the entire process
    Z_world = depth_to_world(D_o, Znear, Zfar)
    world_coords = image_to_world_coordinates(D_o, K_o, Z_world)
    image_coords = world_to_image_coordinates(world_coords, Rt_v, K_v)
    V_virtual = render_virtual_view_optimized(V_o, image_coords)

    return V_virtual






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


N = 5
views = generate_multiple_views(V_original, D_original, K_original, Rt_original, K_virtual, Rt_virtual, Znear, Zfar, N)

# Save each view
for idx, view in enumerate(views):
    output_image_path = f'view_{idx}.png'
    cv2.imwrite(output_image_path, view)
    print(f"View {idx} saved to {output_image_path}")
