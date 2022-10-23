from cryptography.fernet import Fernet

SECRET_KEY = Fernet.generate_key().decode()
print(SECRET_KEY)