"""
core/cbz_handler.py
Extract images from a CBZ file into a temp directory, and repack a
directory of images back into a clean CBZ (no extra metadata).
"""

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple
import pillow_avif

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif"}


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def extract_cbz(cbz_path: str, webtoon_strip_height: int = 0,
                 chunk_height: int = 0, chunk_overlap: int = 0) -> Tuple[str, List[str]]:
    """
    Extract all images from *cbz_path* into a temporary directory.

    Chunking modes (evaluated in priority order):
      1. chunk_height > 0: Stitch all pages into one tall strip, then slice
         into overlapping chunks of *chunk_height* px with *chunk_overlap* px
         shared between consecutive chunks.
      2. webtoon_strip_height > 0: Legacy mode — merge consecutive images
         with similar widths up to the specified height limit (no overlap).

    Returns
    -------
    (tmp_dir, sorted_image_paths)
        tmp_dir        – caller must delete this directory when finished.
        sorted_image_paths – absolute paths to images, sorted naturally.
    """
    cbz_path = Path(cbz_path).resolve()
    if not cbz_path.exists():
        raise FileNotFoundError(f"CBZ not found: {cbz_path}")

    tmp_dir = tempfile.mkdtemp(prefix="cbz_extract_")
    logger.info("Extracting %s → %s", cbz_path.name, tmp_dir)

    try:
        with zipfile.ZipFile(cbz_path, "r") as zf:
            for member in zf.namelist():
                if _is_image(member):
                    zf.extract(member, tmp_dir)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"Not a valid CBZ/ZIP file: {cbz_path}") from exc

    images = sorted(
        (
            str(Path(tmp_dir) / f)
            for f in _walk_images(tmp_dir)
        ),
        key=_natural_sort_key,
    )

    if not images:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"No images found inside CBZ: {cbz_path}")

    # Priority 1: Overlapping chunk mode
    if chunk_height > 0:
        logger.info(
            "Stitch-and-chunk mode: chunk_height=%d, overlap=%d",
            chunk_height, chunk_overlap,
        )
        images = _stitch_and_chunk(tmp_dir, images, chunk_height, chunk_overlap)
    # Priority 2: Legacy webtoon merge (no overlap)
    elif webtoon_strip_height > 0:
        logger.info(f"Merging webtoon slices vertically up to {webtoon_strip_height}px...")
        images = _merge_vertical_strips(tmp_dir, images, webtoon_strip_height)

    logger.info("Extracted %d images (after any merging)", len(images))
    return tmp_dir, images


def _stitch_and_chunk(
    tmp_dir: str,
    image_paths: List[str],
    chunk_height: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Stitch all pages vertically, then slice into overlapping chunks.

    Writes ``chunk_metadata.json`` in *tmp_dir* with global y-offsets for
    each chunk so downstream deduplication can compare bubble positions
    across chunk boundaries.

    Uses a two-pass streaming approach to avoid loading the entire stitched
    image into memory:
      Pass 1 – measure all page dimensions.
      Pass 2 – for each output chunk, open only the pages that contribute
               pixels and paste the relevant rows.
    """
    import json
    from PIL import Image

    if not image_paths:
        return image_paths

    # Clamp overlap to be less than chunk height
    chunk_overlap = max(0, min(chunk_overlap, chunk_height - 100))
    stride = chunk_height - chunk_overlap  # pixels advanced per chunk

    # ── Pass 1: measure ────────────────────────────────────────────────────
    page_dims = []  # list of (width, height, path)
    max_width = 0

    for p in image_paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                page_dims.append((w, h, p))
                max_width = max(max_width, w)
        except Exception as e:
            logger.warning("Cannot read %s for dimensions: %s", p, e)

    if not page_dims:
        return image_paths

    # Recalculate heights based on max_width scaling
    total_height = 0
    page_y_offsets = []  # (y_start, y_end, width, path)
    
    for w, h, p in page_dims:
        if w != max_width and w > 0:
            scale = max_width / w
            scaled_h = int(h * scale)
        else:
            scaled_h = h
            
        page_y_offsets.append((total_height, total_height + scaled_h, w, p))
        total_height += scaled_h

    # If total height is shorter than one chunk, just return as-is
    if total_height <= chunk_height:
        logger.info("Total height %dpx ≤ chunk_height %dpx — no chunking needed.",
                     total_height, chunk_height)
        return image_paths

    # ── Pass 2: slice chunks ──────────────────────────────────────────────
    chunk_paths = []
    chunk_meta = []  # for deduplication
    chunk_idx = 0
    chunk_y = 0

    while chunk_y < total_height:
        chunk_y_end = min(chunk_y + chunk_height, total_height)
        actual_height = chunk_y_end - chunk_y

        # Create canvas for this chunk
        canvas = Image.new("RGB", (max_width, actual_height), (0, 0, 0))

        # Find pages that overlap with [chunk_y, chunk_y_end)
        for page_y_start, page_y_end, page_w, page_path in page_y_offsets:
            # Check for overlap
            overlap_start = max(chunk_y, page_y_start)
            overlap_end = min(chunk_y_end, page_y_end)

            if overlap_start >= overlap_end:
                continue  # no overlap

            try:
                with Image.open(page_path) as page_img:
                    # Resize to max_width if needed
                    if page_img.width != max_width:
                        scale = max_width / page_img.width
                        new_h = int(page_img.height * scale)
                        page_img = page_img.resize(
                            (max_width, new_h), Image.Resampling.LANCZOS
                        )

                    # Crop the relevant rows from this page
                    crop_top = overlap_start - page_y_start
                    crop_bottom = overlap_end - page_y_start
                    strip = page_img.crop((0, crop_top, max_width, crop_bottom))

                    # Paste into canvas at correct position
                    paste_y = overlap_start - chunk_y
                    canvas.paste(strip, (0, paste_y))
            except Exception as e:
                logger.warning("Error pasting page %s into chunk %d: %s",
                               page_path, chunk_idx, e)

        # Save chunk
        chunk_name = f"chunk_{chunk_idx:04d}.png"
        chunk_path = os.path.join(tmp_dir, chunk_name)
        canvas.save(chunk_path, format="PNG")
        chunk_paths.append(chunk_path)

        chunk_meta.append({
            "chunk_index": chunk_idx,
            "y_start": chunk_y,
            "y_end": chunk_y_end,
            "height": actual_height,
            "width": max_width,
            "file": chunk_name,
        })

        logger.debug("  Chunk %d: y=%d→%d (%dpx)",
                      chunk_idx, chunk_y, chunk_y_end, actual_height)

        chunk_idx += 1
        chunk_y += stride

        # If remaining height is completely covered by the previous chunk's overlap, stop!
        if chunk_y + chunk_overlap >= total_height:
            break

    canvas = None  # free memory

    # Write metadata for deduplication
    meta_path = os.path.join(tmp_dir, "chunk_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_height": chunk_height,
            "chunk_overlap": chunk_overlap,
            "total_stitched_height": total_height,
            "stitched_width": max_width,
            "num_chunks": len(chunk_meta),
            "chunks": chunk_meta,
        }, f, indent=2)

    # Clean up original extracted pages (no longer needed)
    for _, _, _, p in page_y_offsets:
        try:
            os.remove(p)
        except OSError:
            pass

    logger.info(
        "Stitched %d pages (%dpx tall) → %d overlapping chunks",
        len(page_dims), total_height, len(chunk_paths),
    )
    return sorted(chunk_paths, key=_natural_sort_key)


def _merge_vertical_strips(tmp_dir: str, image_paths: List[str], max_height: int) -> List[str]:
    from PIL import Image
    if not image_paths or max_height <= 0:
        return image_paths

    merged_paths = []
    current_batch = []
    current_height = 0
    current_width = 0
    out_idx = 1

    def save_batch():
        nonlocal out_idx, current_batch, current_height, current_width
        if not current_batch:
            return
            
        if len(current_batch) == 1:
            # Just move and rename
            img_path = current_batch[0]
            new_path = os.path.join(tmp_dir, f"merged_page_{out_idx:04d}.png")
            shutil.move(img_path, new_path)
            merged_paths.append(new_path)
        else:
            try:
                merged = Image.new("RGB", (current_width, current_height))
                y_offset = 0
                for ipath in current_batch:
                    img = Image.open(ipath)
                    # Resize width to match if slightly different
                    if img.width != current_width:
                        new_h = int(img.height * (current_width / img.width))
                        img = img.resize((current_width, new_h), Image.Resampling.LANCZOS)
                        
                    merged.paste(img, (0, y_offset))
                    y_offset += img.height
                    img.close()
                    os.remove(ipath)
                
                new_path = os.path.join(tmp_dir, f"merged_page_{out_idx:04d}.png")
                merged.save(new_path)
                merged_paths.append(new_path)
            except Exception as e:
                logger.error("Failed to merge batch: %s", e)
                # Fallback: just append original paths if merge fails
                merged_paths.extend(current_batch)
                
        out_idx += 1
        current_batch.clear()
        current_height = 0
        current_width = 0

    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                
            if not current_batch:
                current_batch.append(path)
                current_width = w
                current_height = h
            elif (current_height + h) <= max_height and abs(current_width - w) < 200:
                # Fits within limit and width is roughly the same
                current_batch.append(path)
                current_height += h
            else:
                save_batch()
                current_batch.append(path)
                current_width = w
                current_height = h
        except Exception as e:
            logger.warning("Could not read image %s for metadata: %s", path, e)
            save_batch()
            merged_paths.append(path)

    save_batch()
    return sorted(merged_paths, key=_natural_sort_key)



def repack_cbz(image_dir: str, output_path: str) -> str:
    """
    Pack all images from *image_dir* into a clean CBZ at *output_path*.
    No extra metadata, comments, or hidden files are included.

    Returns the absolute path to the created CBZ.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(
        (p for p in _walk_images(str(image_dir))),
        key=_natural_sort_key,
    )

    if not images:
        raise ValueError(f"No images found in directory: {image_dir}")

    logger.info("Repacking %d images → %s", len(images), output_path.name)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for rel_path in images:
            full_path = image_dir / rel_path
            zf.write(full_path, arcname=rel_path)

    logger.info("Repacked CBZ saved: %s", output_path)
    return str(output_path.resolve())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _walk_images(base_dir: str) -> List[str]:
    """Return relative paths to all image files under base_dir."""
    result = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if _is_image(fname):
                rel = os.path.relpath(os.path.join(root, fname), base_dir)
                result.append(rel)
    return result


def _natural_sort_key(s: str):
    """Sort strings containing numbers in human/natural order."""
    import re
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]
