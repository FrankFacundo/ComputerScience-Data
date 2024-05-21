from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwe
from jose.constants import ALGORITHMS


def generate_and_save_keys():
    # Generate private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Save private key to a file
    with open("private_key.pem", "wb") as private_key_file:
        private_key_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Generate public key
    public_key = private_key.public_key()

    # Save public key to a file
    with open("public_key.pem", "wb") as public_key_file:
        public_key_file.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )


def encrypt_phrase(phrase):
    # Load public key from a file
    with open("public_key.pem", "rb") as public_key_file:
        public_key = public_key_file.read()

    # Encrypt the phrase using JWE
    encrypted_phrase = jwe.encrypt(
        phrase.encode("utf-8"),
        public_key,
        algorithm=ALGORITHMS.RSA_OAEP,
        encryption=ALGORITHMS.A256GCM,
    )

    print("Encrypted phrase:", encrypted_phrase)
    return encrypted_phrase


def decrypt_phrase(encrypted_phrase):
    # Load private key from a file
    with open("private_key.pem", "rb") as private_key_file:
        private_key = private_key_file.read()

    # Decrypt the phrase using JWE
    decrypted_phrase = jwe.decrypt(encrypted_phrase, private_key)

    print("Decrypted phrase:", decrypted_phrase.decode("utf-8"))
    return decrypted_phrase.decode("utf-8")


# Example usage
generate_and_save_keys()
encrypted = encrypt_phrase("This is a secret message.")
decrypt_phrase(encrypted)
