import subprocess

def get_cuda_version_as_number():
    try:
        # Run nvidia-smi and capture the output
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        # Look for the line containing "CUDA Version"
        for line in output.split("\n"):
            if "CUDA Version" in line:
                # Extract the version part after the colon and strip unwanted characters
                version_str = line.split(":")[-1].strip()
                version_number = version_str.split()[0]  # Remove extra symbols or text if present
                return float(version_number)  # Convert to float (e.g., 11.8)
    except Exception as e:
        raise RuntimeError(f"Error fetching CUDA version: {e}")

cuda_version = get_cuda_version_as_number()
print("CUDA Version (as number):", cuda_version)
