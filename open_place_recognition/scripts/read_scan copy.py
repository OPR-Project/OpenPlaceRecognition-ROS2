#!/usr/bin/env python3
import os
import struct

def main():
    # Path to your scan binary file (adjust as needed)
    bin_path = "/home/rover2/Downloads/08_2023-10-11-night/lidar/1697045307522036406.bin"
    
    # Read the entire binary file.
    with open(bin_path, "rb") as f:
        data = f.read()
    
    # Print file size and header (first 16 bytes)
    print("File size:", len(data))
    header = data[:16]
    print("Header (hex):", header.hex())
    
    # Assume the first 16 bytes are header metadata.
    # The rest of the file is raw float32 data.
    float_data = data[16:]
    n_floats = len(float_data) // 4
    print("Number of floats:", n_floats)
    
    # Unpack the data as little-endian float32 values.
    floats = struct.unpack("<" + "f" * n_floats, float_data)
    
    # Assume each point is represented by 4 floats: x, y, z, intensity (or a placeholder)
    points_per_entry = 4
    if n_floats % points_per_entry != 0:
        print("Warning: Total float count is not a multiple of", points_per_entry)
    n_points = n_floats // points_per_entry
    print("Number of points:", n_points)
    print("First 10 floats:", floats[:10])
    
    # Create an ASCII PCD file from these points.
    pcd_filename = "1697045307522036406.pcd"
    with open(pcd_filename, "w") as f:
        # Write PCD header (version 0.7, ASCII format)
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH {}\n".format(n_points))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS {}\n".format(n_points))
        f.write("DATA ascii\n")
        # Write each point (one per line)
        for i in range(n_points):
            idx = i * points_per_entry
            x, y, z, intensity = floats[idx], floats[idx+1], floats[idx+2], floats[idx+3]
            f.write(f"{x} {y} {z} {intensity}\n")
    
    print("PCD file written:", pcd_filename)

if __name__ == '__main__':
    main()
