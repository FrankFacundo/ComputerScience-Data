from cryptography.fernet import Fernet

SECRET_KEY = Fernet.generate_key().decode()
print(SECRET_KEY)
print(len(SECRET_KEY))

### secrets just generate random numbers, it is not a cryptographic lib as Fernet
import secrets

def generate_api_key(length=32):
    if length < 16:
        print("API key length should be at least 16 characters for better security.")
        return None

    # Generate a random API key with the given length
    api_key = secrets.token_urlsafe(length)

    return api_key

api_key = generate_api_key(32)
print("Generated API key:", api_key)
print(len(api_key))
