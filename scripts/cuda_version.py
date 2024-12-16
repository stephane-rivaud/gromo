import subprocess

def get_cuda_version():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        for line in output.split("\n"):
            if "CUDA Version" in line:
                return line.split(":")[-1].strip()
    except Exception as e:
        return f"Error: {e}"

print("CUDA Version:", get_cuda_version())
