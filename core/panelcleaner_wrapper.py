import os
import cv2
import numpy as np
import torch
import onnxruntime as ort

# Point to Panel Cleaner models
ONNX_MODEL_PATH = r"D:\PanelCleaner-Windows-2.11.4_unzip_me\_internal\pcleaner\cache\model\comictextdetector.pt.onnx"
LAMA_MODEL_PATH = r"D:\PanelCleaner-Windows-2.11.4_unzip_me\_internal\pcleaner\cache\model\anime-manga-big-lama.pt"

class PanelCleanerPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load ONNX Text Detector
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        if device == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("WARNING: CUDAExecutionProvider not found in ONNXRuntime. Falling back to CPU for text detection.")
            providers = ['CPUExecutionProvider']
            
        try:
            self.detector_sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
            
        # Load Torch LaMa
        try:
            self.lama_model = torch.jit.load(LAMA_MODEL_PATH, map_location=device)
            self.lama_model.eval()
            self.lama_model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load LaMa model: {e}")

    def detect_text_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the comic text detector.
        Returns a binary mask (H, W) where 255 is text, 0 is background.
        """
        h, w = image.shape[:2]
        
        # Resize preserving aspect ratio into 1024x1024
        scale = 1024 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to 1024x1024 (bottom right padding)
        pad_h = 1024 - new_h
        pad_w = 1024 - new_w
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # Prepare input
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose((2, 0, 1))[np.newaxis, ...]
        
        # Run ONNX inference
        outputs = self.detector_sess.run(None, {'images': input_tensor})
        seg = outputs[1][0, 0] # segmentation logits/probs
        
        # Threshold and create mask
        mask = (seg > 0.3).astype(np.uint8) * 255
        
        # Unpad
        mask = mask[:new_h, :new_w]
        
        # Resize back to original
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Dilation and smoothing (PanelCleaner default processing)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=3)
        return mask

    def inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint image given the mask using LaMa model.
        """
        img_tensor = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Format mask
        mask_tensor = mask.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            # Pad size to multiple of 8 (LaMa requirement)
            h, w = img_tensor.shape[2:]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            
            if pad_h > 0 or pad_w > 0:
                import torch.nn.functional as F
                img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                
            inpainted = self.lama_model(img_tensor, mask_tensor)
            
            if pad_h > 0 or pad_w > 0:
                inpainted = inpainted[:, :, :h, :w]
                
            res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            res = np.clip(res * 255, 0, 255).astype(np.uint8)
            
        return res
        
    def clean_panel(self, image_path: str, output_path: str):
        """
        End-to-end cleaning process for a single image.
        """
        # Read with PIL and convert to numpy to ensure RGB
        from PIL import Image
        img_pil = Image.open(image_path).convert("RGB")
        image = np.array(img_pil)
        
        mask = self.detect_text_mask(image)
        inpainted = self.inpaint_lama(image, mask)
        
        out_pil = Image.fromarray(inpainted)
        out_pil.save(output_path, format="PNG")
        return output_path
