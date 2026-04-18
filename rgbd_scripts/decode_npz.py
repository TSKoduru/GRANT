import cv2
import numpy as np

# Assuming scan_types.py is in the same directory
from scan_types import RGBDFrame, CameraIntrinsics

def load_rgbd_from_npz(filepath: str, intrinsics: CameraIntrinsics) -> RGBDFrame:
    """
    Loads RGB and depth data from a compressed .npz file 
    and packages it into an RGBDFrame.
    """
    try:
        with np.load(filepath) as archive:
            rgb_array = archive['rgb']
            depth_array = archive['depth']
            
            return RGBDFrame(
                rgb=rgb_array,
                depth=depth_array,
                intrinsics=intrinsics,
                arm_mask=None 
            )
            
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        raise
    except KeyError as e:
        print(f"Error: Missing expected key in the .npz archive: {e}")
        raise

def visualize_rgbd(frame: RGBDFrame):
    """
    Converts raw RGBDFrame arrays into displayable 8-bit images 
    and renders them in OpenCV windows.
    """
    # 1. Prepare the RGB Image
    # OpenCV expects BGR format for imshow, but our array is in RGB
    display_rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

    # 2. Prepare the Depth Image
    # Filter out invalid depth pixels (0 meters) so they don't ruin our min/max scaling
    valid_mask = frame.depth > 0
    
    # Initialize an empty 8-bit array for the normalized depth
    normalized_depth = np.zeros_like(frame.depth, dtype=np.uint8)
    
    if np.any(valid_mask):
        # Find the closest and furthest valid points
        min_depth = np.min(frame.depth[valid_mask])
        max_depth = np.max(frame.depth[valid_mask])
        
        # Scale the float meters to a 0-255 range
        scaled = (frame.depth - min_depth) / (max_depth - min_depth + 1e-6) * 255.0
        normalized_depth = np.clip(scaled, 0, 255).astype(np.uint8)

    # Apply a colormap (JET) to make the depth easier to interpret visually
    display_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    
    # Force invalid/zero-depth pixels to be strictly black
    display_depth[~valid_mask] = 0

    # 3. Render the Windows
    cv2.imshow("Reconstructed RGB", display_rgb)
    cv2.imshow("Reconstructed Depth", display_depth)

    print("Windows are open. Press any key while hovering over a window to close them.")
    
    # Wait indefinitely until the user presses a key, then clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Dummy intrinsics required by the RGBDFrame dataclass
    dummy_intrinsics = CameraIntrinsics(
        fx=1000.0, fy=1000.0, 
        cx=640.0, cy=480.0, 
        width=1280, height=960
    )
    
    # Replace with your actual saved file name
    target_file = "scan_frame_1776490226.npz"
    
    try:
        # Load the frame
        frame = load_rgbd_from_npz(target_file, dummy_intrinsics)
        print(f"Successfully loaded {target_file}")
        
        # Reconstruct and display the graphical representation
        visualize_rgbd(frame)
        
    except Exception as e:
        print(f"Failed to load or display frame: {e}")
