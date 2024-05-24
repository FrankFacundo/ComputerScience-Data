import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def generate_and_save_keys():
    # Generate private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    private_key_filepath = os.path.join(os.path.dirname(__file__), "private_key.pem")

    # Save private key to a file
    with open(private_key_filepath, "wb") as private_key_file:
        private_key_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Generate public key
    public_key = private_key.public_key()

    public_key_filepath = os.path.join(os.path.dirname(__file__), "public_key.pem")

    # Save public key to a file
    with open(public_key_filepath, "wb") as public_key_file:
        public_key_file.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )


def encrypt_phrase(phrase):
    # Load public key from a file
    public_key_filepath = os.path.join(os.path.dirname(__file__), "public_key.pem")

    with open(public_key_filepath, "rb") as public_key_file:
        public_key = serialization.load_pem_public_key(public_key_file.read())

    # Encrypt the phrase
    encrypted_phrase = public_key.encrypt(
        phrase.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    print("Encrypted phrase:", encrypted_phrase)
    return encrypted_phrase


def decrypt_phrase(encrypted_phrase):
    # Load private key from a file
    public_key_filepath = os.path.join(os.path.dirname(__file__), "private_key.pem")

    with open(public_key_filepath, "rb") as private_key_file:
        private_key = serialization.load_pem_private_key(
            private_key_file.read(), password=None
        )

    # Decrypt the phrase
    decrypted_phrase = private_key.decrypt(
        encrypted_phrase,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    print("Decrypted phrase:", decrypted_phrase.decode("utf-8"))
    return decrypted_phrase.decode("utf-8")


# Example usage
generate_and_save_keys()
encrypted = encrypt_phrase("This is a secret message.")
decrypt_phrase(encrypted)
