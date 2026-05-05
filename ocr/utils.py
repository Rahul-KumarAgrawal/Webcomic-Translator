import os
import sys
import time
import site

def link_dlls():
    """Fixes 'cudnn64_8.dll not found' by linking required binaries from the cache or search paths."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root, ".dll_paths.cache")
    
    # 1. Try loading from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_paths = [l.strip() for l in f if l.strip()]
            for p in cached_paths:
                if os.path.exists(p):
                    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(p)
                        except Exception: pass
            return
        except Exception: pass

    # 2. Deep scan if cache missing
    search_paths = site.getsitepackages() + sys.path
    found_dirs = []
    
    for s_path in search_paths:
        if not s_path or not os.path.exists(s_path) or not os.path.isdir(s_path): continue
        
        # Link NVIDIA packages
        nv_root = os.path.join(s_path, "nvidia")
        if os.path.exists(nv_root):
            for sub in os.listdir(nv_root):
                bp = os.path.join(nv_root, sub, "bin")
                if os.path.exists(bp): found_dirs.append(bp)
        
        # Link Torch/lib
        torch_lib = os.path.join(s_path, "torch", "lib")
        if os.path.exists(torch_lib): found_dirs.append(torch_lib)

        # Link Paddle libs
        paddle_libs = os.path.join(s_path, "paddle", "libs")
        if os.path.exists(paddle_libs): found_dirs.append(paddle_libs)

    # Apply and Save
    unique_dirs = list(set(found_dirs))
    for p in unique_dirs:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
            except Exception: pass
            
    with open(cache_path, "w") as f:
        f.write("\n".join(unique_dirs))
