import struct

with open("/home/rover2/Downloads/08_2023-10-11-night/lidar/1697045307522036406.bin", "rb") as f:
    data = f.read()

# Print the first 16 bytes in hex for inspection.
print("Header (hex):", data[:16].hex())

# Try interpreting the first 4 bytes as little-endian integer.
try:
    count_le = struct.unpack('<i', data[:4])[0]
    print("Little-endian count:", count_le)
except Exception as e:
    print("Error unpacking as little-endian integer:", e)

# Try interpreting as big-endian.
try:
    count_be = struct.unpack('>i', data[:4])[0]
    print("Big-endian count:", count_be)
except Exception as e:
    print("Error unpacking as big-endian integer:", e)
    
    
import zlib
import struct

# Open the binary scan file.
with open("/home/rover2/.ros/opr_dataset/my_map/scan/node_1_9_scan.bin", "rb") as f:
    data = f.read()

print("Compressed file size:", len(data))
print("Header (hex):", data[:16].hex())

# Decompress using zlib.
try:
    decompressed = zlib.decompress(data)
except Exception as e:
    print("Decompression failed:", e)
    exit(1)

print("Decompressed data size:", len(decompressed))

# (Optional) Interpret the decompressed data as a sequence of float32 values.
# Note: The internal format of scan data is defined by RTAB-Map.
n_floats = len(decompressed) // 4  # Number of 4-byte floats.
try:
    floats = struct.unpack("f" * n_floats, decompressed)
    print("First 10 floats:", floats[:10])
except Exception as e:
    print("Error unpacking floats:", e)


# Interpret the decompressed data as 32-bit floats.
n_floats = len(decompressed) // 4
floats = struct.unpack("f" * n_floats, decompressed)

# Assume each point consists of 4 floats: x, y, z, intensity (or a placeholder)
n_points = n_floats // 4
print("Number of points:", n_points)
print("First 10 floats:", floats[:10])

# Create a PCD file (ASCII format)
pcd_filename = "node_1_9_scan.pcd"
with open(pcd_filename, "w") as f:
    # Write the header for an ASCII PCD file.
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
    # Write each point.
    for i in range(n_points):
        idx = i * 4
        x, y, z, intensity = floats[idx], floats[idx+1], floats[idx+2], floats[idx+3]
        f.write(f"{x} {y} {z} {intensity}\n")

print("PCD file written:", pcd_filename)
