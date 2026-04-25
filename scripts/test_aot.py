import sys
import os
import onnxruntime as ort
import numpy as np
from PIL import Image

def test_aot():
    model_path = r"d:\Translate\Translator\Pipeline Koharu\Inpainting\aot-inpainting\aot.onnx"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        # Try fallback path
        model_path = r"d:\Translate\Translator\models\easyocr\Inpainting\aot-inpainting\aot.onnx"
        if not os.path.exists(model_path):
            print("Fallback also not found.")
            return

    print("Loading ONNX...")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    mask_name = sess.get_inputs()[1].name
    out_name = sess.get_outputs()[0].name

    print(f"Input: {inp_name} ({sess.get_inputs()[0].shape})")
    print(f"Mask: {mask_name} ({sess.get_inputs()[1].shape})")
    print(f"Output: {out_name} ({sess.get_outputs()[0].shape})")

    # Let's generate a test image
    np.random.seed(42)
    img_np = np.random.rand(1, 3, 256, 256).astype(np.float32) * 2 - 1
    mask_np = np.zeros((1, 1, 256, 256), dtype=np.float32)
    mask_np[:, :, 100:150, 100:150] = 1.0 # 1 means masked?

    print("Running with mask = 1.0...")
    out1 = sess.run([out_name], {inp_name: img_np, mask_name: mask_np})[0]
    print(f"Out1 shape: {out1.shape}, min: {out1.min()}, max: {out1.max()}")

    print("Running with mask = 255.0...")
    out2 = sess.run([out_name], {inp_name: img_np, mask_name: mask_np * 255.0})[0]
    print(f"Out2 shape: {out2.shape}, min: {out2.min()}, max: {out2.max()}")

if __name__ == "__main__":
    test_aot()
