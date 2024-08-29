# Depth Image-Based Rendering (DIBR) Project

## Overview

This project implements a Depth Image-Based Rendering (DIBR) algorithm in Python to generate synthesized 2D virtual views from an original 2D image and a corresponding depth map. The generated views exhibit natural perspective changes based on the camera's intrinsic and extrinsic parameters.

## Files

- `dibr.py`: Contains the `DIBR` function for generating a single virtual view.
- `generate_views.py`: Script for generating multiple views using the `DIBR` function.
- `V_original.png`: Original input image.
- `D_original.png`: Depth map corresponding to the original image.
- `view_0.png`, `view_1.png`, ..., `view_N.png`: Generated virtual views.

## Usage

1. **Install Dependencies:**
   - Ensure you have Python installed with the following libraries:
     ```bash
     pip install numpy opencv-python matplotlib scipy
     ```

2. **Run the `generate_views.py` script:**
   - Use the provided script to generate multiple views by running:
     ```bash
     python generate_views.py
     ```

3. **Output:**
   - The synthesized virtual views will be saved as PNG images in the current directory.

## Assumptions

- The depth map (`D_original.png`) is assumed to have values normalized between 0 and 255.
- Camera parameters (intrinsic and extrinsic) must be provided in the correct format.

## Design Choices

- The `griddata` function is used for interpolating pixel colors to handle cracks and disocclusions.
- `cv2.inpaint` is used to fill in any remaining black regions after interpolation.

## License

This project is licensed under the MIT License.
