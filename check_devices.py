import GPUtil

# Get the GPU devices
gpus = GPUtil.getGPUs()

# Print available GPU devices
if len(gpus) == 0:
    print("No GPU available.")
else:
    print("Available GPUs:")
    for gpu in gpus:
        print(f"ID: {gpu.id}, GPU: {gpu.name}")
