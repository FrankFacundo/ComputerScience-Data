from jose import jwt
from jose.constants import ALGORITHMS


def sign_jwt(payload, private_key_path):
    # Load private key from a file
    with open(private_key_path, "rb") as private_key_file:
        private_key = private_key_file.read()

    # Sign the JWT
    token = jwt.encode(payload, private_key, algorithm=ALGORITHMS.RS256)
    print("Signed JWT:", token)
    return token


def verify_jwt(token, public_key_path):
    # Load public key from a file
    with open(public_key_path, "rb") as public_key_file:
        public_key = public_key_file.read()

    # Verify the JWT
    try:
        payload = jwt.decode(token, public_key, algorithms=[ALGORITHMS.RS256])
        print("Verified JWT payload:", payload)
        return payload
    except jwt.JWTError as e:
        print("JWT verification failed:", e)
        return None


payload = {"sub": "1234567890", "name": "John Doe", "admin": True}

# Sign the JWT
private_key_path = "private_key.pem"
token = sign_jwt(payload, private_key_path)

# Verify the JWT
public_key_path = "public_key.pem"
verified_payload = verify_jwt(token, public_key_path)
