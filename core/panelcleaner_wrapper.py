import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
import logging

logger = logging.getLogger("core.panelcleaner")
if not logger.handlers:
    import sys
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[PanelCleaner] %(message)s"))
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Relative paths to models in the workspace
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_MODEL_PATH = os.path.join(_ROOT, "models", "detection", "comictextdetector.pt.onnx")
LAMA_MODEL_PATH = os.path.join(_ROOT, "models", "inpainting", "anime-manga-big-lama.pt")

# Fallback to Pipeline Koharu if main models folder is missing them
if not os.path.exists(ONNX_MODEL_PATH):
    ONNX_MODEL_PATH = os.path.join(_ROOT, "Pipeline Koharu", "Detection and Layout", "detector.onnx")
if not os.path.exists(LAMA_MODEL_PATH):
    # Try AOT model from Koharu as a fallback if LaMa isn't found
    LAMA_MODEL_PATH = os.path.join(_ROOT, "Pipeline Koharu", "Inpainting", "aot-inpainting", "aot_traced.pt")

# Persistent instances to avoid re-loading for every page
_DETECTOR_SESS = None
_LAMA_MODEL = None

class PanelCleanerPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        global _DETECTOR_SESS, _LAMA_MODEL
        
        # 1. Text Detector Session
        if _DETECTOR_SESS is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            
            # Search for detector.onnx
            possible_det = [
                ONNX_MODEL_PATH,
                os.path.join(_ROOT, "Pipeline Koharu", "Detection and Layout", "detector.onnx"),
                os.path.join(_ROOT, "models", "detector.onnx")
            ]
            actual_det = next((p for p in possible_det if os.path.exists(p)), None)
            
            if not actual_det:
                logger.error("[PanelCleaner] detector.onnx not found in any expected location!")
                raise FileNotFoundError("detector.onnx missing")
                
            logger.info(f"[VRAM] Loading PanelCleaner Detector: {os.path.basename(actual_det)}")
            _DETECTOR_SESS = ort.InferenceSession(actual_det, sess_options, providers=providers)
            
        self.detector_sess = _DETECTOR_SESS

        # 2. Inpainting Model (LaMa or AOT)
        if _LAMA_MODEL is None:
            possible_inp = [
                os.path.join(_ROOT, "models", "inpainting", "mayo_panel_cleaner.pt"),
                LAMA_MODEL_PATH,
                # Fix: Check both hyphen and underscore versions
                os.path.join(_ROOT, "Pipeline Koharu", "Inpainting", "lama-manga", "lama_manga.pt"),
                os.path.join(_ROOT, "Pipeline Koharu", "Inpainting", "lama_manga", "lama_manga.pt"),
                os.path.join(_ROOT, "Pipeline Koharu", "Inpainting", "aot-inpainting", "aot_traced.pt"),
                os.path.join(_ROOT, "models", "inpainting", "anime-manga-big-lama.pt"),
                os.path.join(_ROOT, "models", "lama.pt")
            ]
            actual_inp = next((p for p in possible_inp if os.path.exists(p)), None)
            
            if not actual_inp:
                logger.error("[PanelCleaner] Inpainting model (LaMa/AOT) not found!")
                raise FileNotFoundError("Inpainting model missing")
                
            model_name = os.path.basename(actual_inp)
            if "mayo" in model_name.lower():
                logger.info(f"[VRAM] Loading High-Quality Mayo Inpainter: {model_name}")
            else:
                logger.info(f"[VRAM] Loading PanelCleaner Inpainter: {model_name}")
            
            _LAMA_MODEL = torch.jit.load(actual_inp, map_location=device)
            _LAMA_MODEL.eval()
            _LAMA_MODEL.to(device)
            
        self.lama_model = _LAMA_MODEL

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
        
        # Newer RT-DETR models require original target sizes
        orig_target_sizes = np.array([[1024, 1024]], dtype=np.int64)
        
        # Run ONNX inference
        try:
            outputs = self.detector_sess.run(None, {
                'images': input_tensor,
                'orig_target_sizes': orig_target_sizes
            })
        except Exception:
            # Fallback for older models that only take 'images'
            outputs = self.detector_sess.run(None, {'images': input_tensor})
            
        # Get segmentation output (usually at index 1)
        seg_output = outputs[1]
        
        # Robust indexing: handle both [1, 1, 1024, 1024] and flatter shapes
        if len(seg_output.shape) == 4:
            seg = seg_output[0, 0]
        elif len(seg_output.shape) == 3:
            seg = seg_output[0]
        else:
            seg = seg_output # Already 2D or 1D
            
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

    def _detect_model_call_order(self):
        """
        Probe the TorchScript model once with a dummy tensor to discover
        which argument order it expects: (img, mask) or (mask, img) or unified 4ch.
        Caches the result so we only probe once per session.
        
        NOTE: LaMa models have 3 downsampling layers + FFT blocks, so the probe
        must use at least 64x64 spatial dimensions. We use 256x256 to be safe.
        """
        if hasattr(self, '_call_order'):
            return self._call_order

        # Must be large enough for LaMa's architecture (3 downsamples + FFT)
        probe_size = 256
        dummy_img = torch.randn(1, 3, probe_size, probe_size, device=self.device)
        dummy_mask = torch.zeros(1, 1, probe_size, probe_size, device=self.device)

        # Strategy A: (img, mask) — standard order (mayo_panel_cleaner.pt uses this)
        try:
            with torch.inference_mode():
                self.lama_model(dummy_img, dummy_mask)
            self._call_order = 'img_mask'
            logger.info("[PanelCleaner] Model call order detected: (image, mask)")
            return self._call_order
        except Exception:
            pass

        # Strategy B: (mask, img) — flipped order
        try:
            with torch.inference_mode():
                self.lama_model(dummy_mask, dummy_img)
            self._call_order = 'mask_img'
            logger.info("[PanelCleaner] Model call order detected: (mask, image)")
            return self._call_order
        except Exception:
            pass

        # Strategy C: unified 4-channel input
        try:
            unified = torch.cat([dummy_img, dummy_mask], dim=1)
            with torch.inference_mode():
                self.lama_model(unified)
            self._call_order = 'unified'
            logger.info("[PanelCleaner] Model call order detected: unified 4ch input")
            return self._call_order
        except Exception:
            pass

        # Fallback — standard (image, mask) order is most common for traced LaMa
        self._call_order = 'img_mask'
        logger.warning("[PanelCleaner] Could not detect model call order, defaulting to (image, mask)")
        return self._call_order

    def _run_model(self, img_t, mask_t):
        """Call the model with the previously detected argument order."""
        order = self._detect_model_call_order()
        if order == 'img_mask':
            return self.lama_model(img_t, mask_t)
        elif order == 'mask_img':
            return self.lama_model(mask_t, img_t)
        else:  # unified
            return self.lama_model(torch.cat([img_t, mask_t], dim=1))

    def inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint image given the mask using LaMa/AOT model.
        """
        # 1. Pre-process image: fill masked areas with neutral color to "help" the AI
        h, w = image.shape[:2]
        
        # Binary mask
        mask_bin = (mask > 127).astype(np.uint8) * 255
        mask_bool = mask_bin > 0
        
        # Fill with median color of the background to provide a stable baseline for the AI
        img_for_inp = image.copy()
        if np.any(mask_bool):
            median_color = np.median(image[~mask_bool], axis=0) if np.any(~mask_bool) else [255, 255, 255]
            img_for_inp[mask_bool] = median_color

        # Convert to tensors [0, 1]
        img_tensor = torch.from_numpy(img_for_inp).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        # LaMa has 3 stride-2 downsample/upsample layers → needs dims divisible by 8
        _, _, th, tw = img_tensor.shape
        pad_h = (8 - th % 8) % 8
        pad_w = (8 - tw % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.inference_mode():
            # Run model with the correct detected argument order
            inpainted = self._run_model(img_tensor, mask_tensor)
            
            # Crop padding back off
            if pad_h > 0 or pad_w > 0:
                inpainted = inpainted[:, :, :th, :tw]

            # Handle output variations (Some models return (img, mask), some return just img)
            if isinstance(inpainted, (list, tuple)):
                inpainted = inpainted[0]

            # Robust Range Normalization
            res = inpainted.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # If output is mostly zero or extremely small, it might expect 0-255 input
            if np.max(res) < 0.05:
                inpainted = self._run_model(img_tensor * 255.0, mask_tensor)
                if isinstance(inpainted, (list, tuple)): inpainted = inpainted[0]
                if pad_h > 0 or pad_w > 0:
                    inpainted = inpainted[:, :, :th, :tw]
                res = inpainted.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

            # Final mapping to 0-255 uint8
            if np.max(res) <= 1.05:
                if np.min(res) < -0.1:
                    res = (res + 1) / 2 # Handle [-1, 1] range
                res = res * 255.0
            
            res = np.clip(res, 0, 255).astype(np.uint8)
            
            # Resize back to original if needed
            if res.shape[0] != h or res.shape[1] != w:
                res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
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

    def unload(self):
        """Free GPU memory by unloading models."""
        if hasattr(self, 'lama_model'):
            del self.lama_model
        if hasattr(self, 'detector_sess'):
            del self.detector_sess
        if self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass
