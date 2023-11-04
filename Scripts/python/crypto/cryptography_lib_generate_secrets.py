import os

from cryptography.fernet import Fernet

dirpath_secret = os.getenv('SECRET_DIRPATH')
secret_key_path = os.getenv('SECRET_KEY_PATH')


# Load the key from the file
print(secret_key_path)
with open(secret_key_path, 'rb') as key_file:
    key = key_file.read()

# Initialize the Fernet class with the key
cipher_suite = Fernet(key)

# Your sensitive data
data = "test"

# Encrypt the data
encrypted_data = cipher_suite.encrypt(data.encode())

# You can now store this encrypted data in a file or elsewhere
with open(os.path.join(dirpath_secret, 'test.bin'), 'wb') as data_file:
    data_file.write(encrypted_data)
