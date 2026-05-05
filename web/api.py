import os
import base64
import httpx
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from autodetect.manager import DetectorManager

app = FastAPI(title="Language Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = DetectorManager()

class SetEngineRequest(BaseModel):
    engine: str

class DetectUrlRequest(BaseModel):
    url: str

@app.get("/autodetect/engines")
async def get_engines(x_api_key: Optional[str] = Header(None)):
    # If header provided, temporarily set it in env for list_engines to see it as configured
    # Actually, manager.list_engines() checks os.environ.
    # We can just return the list and let the UI handle the "is_configured" status based on its own localStorage if needed.
    return manager.list_engines()

@app.post("/autodetect/set-engine")
async def set_engine(req: SetEngineRequest):
    if manager.set_active(req.engine):
        return {"status": "success", "active": req.engine}
    raise HTTPException(status_code=404, detail="Engine not found")

@app.get("/autodetect/test/{engine}")
async def test_engine(engine: str, x_api_key: Optional[str] = Header(None)):
    if x_api_key:
        os.environ[f"{engine.upper()}_API_KEY"] = x_api_key
    
    success = manager.test_engine(engine)
    return {"status": "success" if success else "failed", "engine": engine}

@app.post("/autodetect/detect")
async def detect(
    file: Optional[UploadFile] = File(None),
    data: Optional[DetectUrlRequest] = Body(None),
    x_api_key: Optional[str] = Header(None)
):
    active_engine = manager.get_active()
    if not active_engine:
        engines = manager.list_engines()
        if engines:
            active_engine = manager.engines[engines[0]['name']]
        else:
            raise HTTPException(status_code=400, detail="No detection engine available.")

    if x_api_key:
        os.environ[f"{active_engine.name.upper()}_API_KEY"] = x_api_key

    image_bytes = None
    if file:
        filename = file.filename.lower()
        if filename.endswith(".cbz"):
            # Handle CBZ: Extract pages 4 and 5
            import zipfile
            import io
            content = await file.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # Filter for image files and sort them
                exts = ('.jpg', '.jpeg', '.png', '.webp')
                images = sorted([f for f in z.namelist() if f.lower().endswith(exts)])
                
                if len(images) >= 4:
                    # Page 4 is index 3
                    with z.open(images[3]) as f:
                        image_bytes = f.read()
                elif images:
                    with z.open(images[0]) as f:
                        image_bytes = f.read()
                else:
                    raise HTTPException(status_code=400, detail="No images found in CBZ")
        else:
            image_bytes = await file.read()
    elif data and data.url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(data.url)
            if resp.status_code == 200:
                image_bytes = resp.content
            else:
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided")

    result = active_engine.detect(image_bytes)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
