import subprocess

def bytes_to_megabytes(num_bytes):
    return num_bytes / (1024.0 * 1024.0)

def exec_command(command, shell=False, verbose=False):
    if shell:
        print(f"command to execute: {command}")
    else:
        print(f"command to execute: {' '.join(command)}")
    proc = subprocess.Popen(command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, shell=shell,
                        universal_newlines=True)
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
        if output and verbose:
            print(output.strip())

    # Read and print stderr
    stderr = proc.stderr.read().strip()
    if stderr:
        print(f"Error: {stderr}")

    return
