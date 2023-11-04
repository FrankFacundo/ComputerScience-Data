import os
import subprocess

from cryptography.fernet import Fernet


def get_password():
    secret_key_path = os.getenv('SECRET_KEY_PATH')
    secret_password = os.getenv('SECRET_PASSWORD')

    with open(secret_key_path, 'rb') as key_file:
        key = key_file.read()
    cipher_suite = Fernet(key)

    with open(secret_password, 'rb') as data_file:
        encrypted_pw = data_file.read()

    password = cipher_suite.decrypt(encrypted_pw).decode()
    return password


def execute_command(command, use_sudo=False):
    try:
        if use_sudo:
            sudo_password = get_password()
            # The '-S' flag allows us to pass the password to sudo via stdin
            command = f"echo {sudo_password} | sudo -S {command}"

        process = subprocess.run(command,
                                 shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 check=True)

        if process.returncode == 0:
            return process.stdout
        else:
            return process.stderr
    except Exception as e:
        print(f"An error occurred: {e}")
