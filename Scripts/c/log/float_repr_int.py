import struct

# Your 32-bit integer
int_value = 3149223190
int_value = 1073900465

# # Convert the integer to bytes, assuming it's in big-endian format
# bytes_value = int_value.to_bytes(4, 'big')
# print("Bytes in binary:", ' '.join(format(byte, '08b') for byte in bytes_value))


# # Unpack the bytes to a float
# float_value_big = struct.unpack('>f', bytes_value)[0]
# float_value_little = struct.unpack('<f', bytes_value)[0]

# print("The float representation (big endian) is:", float_value_big)
# print("The float representation (little endian) is:", float_value_little)


float_value_big = struct.unpack('>f', struct.pack('>I', int_value))[0]
float_value_little = struct.unpack('<f', struct.pack('>I', int_value))[0]

print("The float representation (big endian) is:", float_value_big)
print("The float representation (little endian) is:", float_value_little)
