import OpenEXR
import Imath
import numpy as np

def read_exr_channels(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get the header to find out the channels
    header = exr_file.header()
    channels = header['channels'].keys()
    
    # Print available channels
    print("Available channels:", channels)
    
    # Read the first pixel (0, 0) from each channel
    pixel_values = {}
    for channel in channels:
        # Read channel data as a string
        channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        # Convert to numpy array
        channel_data_array = np.frombuffer(channel_data, dtype=np.float32)
        # Extract the first pixel value
        pixel_values[channel] = channel_data_array[0]
    
    # Close the EXR file
    exr_file.close()
    
    # Print the first pixel values for all channels
    for channel, value in pixel_values.items():
        print(f"Channel: {channel}, Value: {value}")
# Example usage
file_path = "C:/Files/CGProject/temp/frame_0950/Mogwai.ReflectMapGen.diffuse.1.exr"
read_exr_channels(file_path)
