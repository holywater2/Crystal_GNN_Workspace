
def print_available_gpu():
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("-----------------------------------------------")
        print(f"[I] Number of available GPUs: {device_count}")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i+1}: {gpu_name}")
        print("-----------------------------------------------")
    else:
        print("[I] No GPUs available.")
        
        
# os.environ["CUDA_VISIBLE_DEVICES"]= args.GPU

        
if __name__ == "__main__":
    print_available_gpu()