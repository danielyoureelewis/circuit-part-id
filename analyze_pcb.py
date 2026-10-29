import io
import base64
import json
import re
import sys
import os
import math
import time
from litellm import completion
from PIL import Image

def call_llm(messages, timeout=1200, max_retries=3):
    """Call the LLM with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            resp = completion(
                model="openai/qwen-vision",
                api_base="http://127.0.0.1:4000/v1",
                api_key="",
                messages=messages,
                response_format={"type": "json_object"},
                timeout=timeout
            )
            return resp
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"    [!] LLM call failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

def check_overlap(boxA, boxB):
    """Calculates IoU for deduplication. Uses center-distance heuristic for edge cases."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    if boxAArea + boxBArea - interArea == 0:
        return 1.0

    iou = interArea / float(boxAArea + boxBArea - interArea)

    if iou < 0.1:
        cx1, cy1 = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
        cx2, cy2 = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
        dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        avg_size = (math.sqrt(boxAArea) + math.sqrt(boxBArea)) / 2
        if dist < avg_size * 0.15:
            return 0.5

    return iou

def get_image_tiles(image_path, target_dim=1024, max_tile_dim=1024, overlap_ratio=0.35):
    """Tile-based detection used only for small targets like debug pins/headers."""
    try:
        pil_image = Image.open(image_path)
        if pil_image.mode in ('RGBA', 'P'):
            pil_image = pil_image.convert('RGB')

        img_width, img_height = pil_image.size

        cols = max(1, math.ceil(img_width / target_dim))
        rows = max(1, math.ceil(img_height / target_dim))

        print(f"[*] Tiling image ({img_width}x{img_height}) into a {cols}x{rows} grid with {int(overlap_ratio*100)}% overlap.")

        base_tw = img_width / cols
        base_th = img_height / rows

        overlap_x = base_tw * overlap_ratio
        overlap_y = base_th * overlap_ratio

        tiles = []
        for row in range(rows):
            for col in range(cols):
                left = int(max(0, col * base_tw - overlap_x))
                upper = int(max(0, row * base_th - overlap_y))
                right = int(min(img_width, (col + 1) * base_tw + overlap_x))
                lower = int(min(img_height, (row + 1) * base_th + overlap_y))

                tile_img = pil_image.crop((left, upper, right, lower))
                orig_w, orig_h = tile_img.size

                tile_img.thumbnail((max_tile_dim, max_tile_dim), Image.Resampling.LANCZOS)
                scaled_w, scaled_h = tile_img.size
                scale_x = orig_w / scaled_w
                scale_y = orig_h / scaled_h

                buffered = io.BytesIO()
                tile_img.save(buffered, format="JPEG", quality=85)
                b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                tiles.append({
                    "b64": b64,
                    "offset_x": left,
                    "offset_y": upper,
                    "orig_tw": orig_w,
                    "orig_th": orig_h,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "scaled_w": scaled_w,
                    "scaled_h": scaled_h
                })
        return tiles, img_width, img_height
    except Exception as e:
        print(f"[!] Error tiling image: {e}")
        return [], 0, 0

def encode_lowres(image_path, max_dimension=1024):
    """Resize image for the coarse Pass 1 detection."""
    try:
        pil_image = Image.open(image_path)
        if pil_image.mode in ('RGBA', 'P'):
            pil_image = pil_image.convert('RGB')
        pil_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        resized_width, resized_height = pil_image.size

        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode('utf-8'), resized_width, resized_height
    except Exception as e:
        print(f"[!] Error encoding low-res image: {e}")
        return None, 0, 0

def encode_crop(pil_img):
    """Encode a high-res cropped region for Pass 2."""
    if pil_img.mode in ('RGBA', 'P'):
        pil_img = pil_img.convert('RGB')
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_interactive_pcb(image_path, output_html="interactive_board.html", crop_dir="components"):
    os.makedirs(crop_dir, exist_ok=True)

    pil_master = Image.open(image_path)
    if pil_master.mode in ('RGBA', 'P'):
        pil_master = pil_master.convert('RGB')
    img_width, img_height = pil_master.size

    with open(image_path, "rb") as image_file:
        b64_full = base64.b64encode(image_file.read()).decode('utf-8')
    mime_type = "image/png" if image_path.lower().endswith('.png') else "image/jpeg"
    image_data_uri = f"data:{mime_type};base64,{b64_full}"

    all_targets = []

    # =========================================================================
    # PASS 1: COARSE DETECTION — Full-board low-res scan for ICs/chips
    # =========================================================================
    print(f"\n[=== PASS 1: COARSE DETECTION (full board, low-res) ===]")
    print(f"[*] Image size: {img_width}x{img_height}")

    b64_lowres, lr_w, lr_h = encode_lowres(image_path)
    if not b64_lowres:
        return

    coarse_prompt = """
    Locate ALL integrated circuits on this full PCB image: MCUs, SoCs, FPGAs, flash, memory, voltage regulators,
    op-amps, power management ICs, transceivers, logic chips — ANY physical IC package.
    The image dimensions are {img_w}x{img_h} pixels.

    STRICT RULES:
    - Box EVERY physical IC package you can see, regardless of function.
    - DO NOT box debug pins, headers, or small pin rows (we handle those separately in Pass 1b).
    - DO NOT box empty PCB, traces, solder points, silkscreen, capacitors, resistors, connectors, or power jacks.
    - Be thorough — miss nothing. It's better to over-detect than to miss a chip.
    - Bounding box must tightly surround the visible package.

    For each detected component, return:
    1. "confidence": 0.0 to 1.0 how certain this is a real component.
    2. "box": [xmin, ymin, xmax, ymax] in ACTUAL PIXELS (0-{img_w} x, 0-{img_h} y).

    Output ONLY a valid JSON object with an array named "components".
    """

    coarse_prompt = coarse_prompt.format(img_w=lr_w, img_h=lr_h)

    try:
        print(f"[*] Sending low-res image ({lr_w}x{lr_h}) to LLM for coarse detection...")
        response = call_llm([{
            "role": "user",
            "content": [
                {"type": "text", "text": coarse_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_lowres}"}}
            ]
        }], timeout=1800)

        llm_output = response.choices[0].message.content
        clean_json = re.sub(r"```json\s*", "", llm_output)
        clean_json = re.sub(r"```\s*", "", clean_json).strip()
        data = json.loads(clean_json)
        coarse_components = data.get("components", [])
        print(f"[*] Coarse pass found {len(coarse_components)} potential IC targets.")

        scale_x = img_width / lr_w
        scale_y = img_height / lr_h

        for part in coarse_components:
            confidence = part.get("confidence", 0.5)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5
            if confidence < 0.5:
                continue

            box = part.get("box")
            if not box or len(box) != 4:
                continue

            # Coarse coords are in low-res pixels, scale to original image
            gl = int(box[0] * scale_x)
            gu = int(box[1] * scale_y)
            gr = int(box[2] * scale_x)
            glw = int(box[3] * scale_y)

            if (gr - gl) < 20 or (glw - gu) < 20:
                continue

            gl = max(0, min(gl, img_width))
            gr = max(0, min(gr, img_width))
            gu = max(0, min(gu, img_height))
            glw = max(0, min(glw, img_height))

            part["global_box"] = [gl, gu, gr, glw]
            all_targets.append(part)

    except Exception as e:
        print(f"[!] Coarse detection failed: {e}")

    # =========================================================================
    # PASS 1b: TILE-BASED DEBUG PIN/HEADER DETECTION
    # =========================================================================
    print(f"\n[=== PASS 1b: TILE-BASED DEBUG PIN DETECTION ===]")

    tiles, _, _ = get_image_tiles(image_path)

    debug_prompt_template = """
    Locate ONLY debug interface pins and headers in this circuit board section:
    - UART pins (TX, RX, CTS, RTS, GND)
    - JTAG header pins (TCK, TMS, TDI, TDO, TRST, RTCK)
    - SWD header pins (SWCLK, SWDIO, GND)
    - Any other clearly labeled debug/test-point headers

    The image dimensions are {img_w}x{img_h} pixels.

    STRICT RULES:
    - ONLY box debug pins and headers. Do NOT box ICs, chips, capacitors, resistors, or other components.
    - DO NOT box large chips that were already detected in the coarse pass.
    - If you see no debug pins, return an empty array.
    - Bounding box must tightly surround the pin row or header area.

    For each detected debug interface, return:
    1. "confidence": 0.0 to 1.0
    2. "box": [xmin, ymin, xmax, ymax] in ACTUAL PIXELS (0-{img_w} x, 0-{img_h} y).

    Output ONLY a valid JSON object with an array named "components".
    """

    for i, tile in enumerate(tiles):
        print(f"[*] Scanning Tile {i+1}/{len(tiles)} for debug pins...")
        tile_data_uri = f"data:image/jpeg;base64,{tile['b64']}"
        prompt = debug_prompt_template.format(img_w=tile["scaled_w"], img_h=tile["scaled_h"])

        try:
            response = call_llm([{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": tile_data_uri}}
                ]
            }], timeout=1200)

            llm_output = response.choices[0].message.content
            clean_json = re.sub(r"```json\s*", "", llm_output)
            clean_json = re.sub(r"```\s*", "", clean_json).strip()
            data = json.loads(clean_json)
            debug_components = data.get("components", [])
            print(f"    Found {len(debug_components)} debug targets in Tile {i+1}.")

            for part in debug_components:
                confidence = part.get("confidence", 0.5)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.5
                if confidence < 0.5:
                    continue

                box = part.get("box")
                if not box or len(box) != 4:
                    continue

                gl = int(tile["offset_x"] + box[0] * tile["scale_x"])
                gu = int(tile["offset_y"] + box[1] * tile["scale_y"])
                gr = int(tile["offset_x"] + box[2] * tile["scale_x"])
                glw = int(tile["offset_y"] + box[3] * tile["scale_y"])

                if (gr - gl) < 10 or (glw - gu) < 10:
                    continue

                gl = max(0, min(gl, img_width))
                gr = max(0, min(gr, img_width))
                gu = max(0, min(gu, img_height))
                glw = max(0, min(glw, img_height))

                part["global_box"] = [gl, gu, gr, glw]
                all_targets.append(part)

        except Exception as e:
            print(f"    [!] Error scanning Tile {i+1}: {e}")

    # Dedup coarse + debug targets
    if len(all_targets) > 1:
        print(f"\n[*] Deduplicating {len(all_targets)} combined targets...")
        deduped = []
        for part in all_targets:
            is_dup = False
            best_idx = None
            best_score = 0
            for idx, existing in enumerate(deduped):
                overlap = check_overlap(part["global_box"], existing["global_box"])
                if overlap > 0.25 and overlap > best_score:
                    best_score = overlap
                    best_idx = idx
            if best_idx is not None:
                is_dup = True
                existing = deduped[best_idx]
                bp = part["global_box"]
                be = existing["global_box"]
                existing["global_box"] = [
                    min(bp[0], be[0]), min(bp[1], be[1]),
                    max(bp[2], be[2]), max(bp[3], be[3])
                ]
            if not is_dup:
                deduped.append(part)
        all_targets = deduped
        print(f"[*] Deduped to {len(all_targets)} unique targets.")

    # =========================================================================
    # PASS 2: HIGH-RES OCR & ANALYSIS on each target
    # =========================================================================
    print(f"\n[=== PASS 2: HIGH-RES OCR & ANALYSIS ===]")

    analyze_prompt = """
    Act as an expert hardware reverse engineer. Look at this close-up of a single PCB component.

    FIRST, read any text visible on the component's package (OCR).
    Then, using the text and visual package type, identify the specific part.

    CRITICAL RULE: Do NOT default to "SPI Flash". Most small chips are actually
    Voltage Regulators (LDOs), Power MOSFETs, Op-Amps, or generic logic gates.
    Only classify as SPI Flash if the part number confirms it (e.g., W25Q, MX25, GD25).

    Return a JSON object containing:
    1. "ocr_text": Any text you read from the component's surface.
    2. "part_name": The specific part number (e.g., 'AMS1117', 'LM358', 'Winbond W25Q64').
    3. "component_class": Classify as ONE of: processor, memory, flash, debug_header, transceiver, regulator, pmic, opamp, logic, passive, unknown.
    4. "description": Explain its function.
    5. "datasheet_link": Google search URL for the datasheet.
    """

    analyzed_components = []
    crop_idx = 0

    for idx, part in enumerate(all_targets):
        gl, gu, gr, glw = part["global_box"]
        gl = max(0, min(gl, img_width))
        gr = max(0, min(gr, img_width))
        gu = max(0, min(gu, img_height))
        glw = max(0, min(glw, img_height))

        if gr <= gl or glw <= gu:
            continue

        # Crop high-res region with padding
        padding_x = max(5, int((gr - gl) * 0.15))
        padding_y = max(5, int((glw - gu) * 0.15))
        crop_left = max(0, gl - padding_x)
        crop_upper = max(0, gu - padding_y)
        crop_right = min(img_width, gr + padding_x)
        crop_lower = min(img_height, glw + padding_y)

        chip_crop = pil_master.crop((crop_left, crop_upper, crop_right, crop_lower))

        # Scale crop for LLM (keep high quality but limit size)
        analysis_crop = chip_crop.copy()
        analysis_crop.thumbnail((512, 512), Image.Resampling.LANCZOS)
        crop_b64 = encode_crop(analysis_crop)

        try:
            res = call_llm([{
                "role": "user",
                "content": [
                    {"type": "text", "text": analyze_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}}
                ]
            }], timeout=1200)

            p2_clean = re.sub(r"```json\s*", "", res.choices[0].message.content)
            p2_clean = re.sub(r"```\s*", "", p2_clean).strip()
            chip_data = json.loads(p2_clean)

            ocr_text = chip_data.get("ocr_text", "")
            part_name = chip_data.get("part_name", f"part_{idx}")
            comp_class = chip_data.get("component_class", "unknown")

            print(f"[*] Target {idx+1}: OCR -> '{ocr_text}'  [{comp_class.upper()}] {part_name}")

            # Save high-res crop for all chips
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', part_name)
            chip_crop.convert('RGB').save(f"{crop_dir}/{crop_idx:02d}_{safe_name}.jpg")
            crop_idx += 1

            chip_data["global_box"] = [gl, gu, gr, glw]
            chip_data["component_class"] = comp_class
            analyzed_components.append(chip_data)

        except Exception as e:
            print(f"    [!] Analysis error: {e}")

    from collections import Counter
    class_counts = Counter(c.get("component_class", "unknown") for c in analyzed_components)
    summary = ", ".join(f"{v} {k}" for k, v in sorted(class_counts.items(), key=lambda x: -x[1]))
    print(f"\n[*] Final result: {len(analyzed_components)} ICs ({summary}).")

    # =========================================================================
    # GENERATE HTML
    # =========================================================================
    print(f"[*] Generating interactive HTML...")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PCB Reverse Engineering Analysis</title>
<style>
  body {{ 
    display: flex; 
    align-items: flex-start;
    font-family: sans-serif; 
    background: #111; 
    color: #eee; 
    padding: 20px; 
    overflow-y: scroll;
  }}
  .image-container {{ position: relative; display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.8); }}
  .pcb-image {{ max-width: 1000px; height: auto; display: block; }}

  .hotspot {{
    position: absolute;
    border: 2px solid;
    background: transparent;
    cursor: pointer;
    box-sizing: border-box;
  }}
  .hotspot:hover {{
    box-shadow: 0 0 12px;
  }}

  .side-panel {{
    width: 380px; 
    min-width: 380px;
    flex-shrink: 0;
    padding: 25px; background: #1a1a1a;
    margin-left: 30px; border-radius: 8px; border: 1px solid #333;
    box-shadow: 0 4px 15px rgba(0,0,0,0.8);
  }}
  .side-panel h2 {{ margin-top: 0; color: #ff4444; border-bottom: 1px solid #333; padding-bottom: 10px; font-size: 1.4em; }}
  .description {{ line-height: 1.5; font-size: 1.05em; }}
  .ocr-text {{ color: #aaa; font-family: monospace; font-size: 0.9em; margin-top: 8px; word-break: break-all; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.8em; margin-top: 6px; font-weight: bold; text-transform: uppercase; }}
  .datasheet-btn {{
    display: inline-block; margin-top: 20px; padding: 12px 18px;
    background: #ff4444; color: #fff; text-decoration: none; border-radius: 5px;
    font-weight: bold; transition: background 0.2s;
  }}
  .datasheet-btn:hover {{ background: #cc0000; }}
  .placeholder-text {{ color: #666; font-style: italic; }}
</style>
</head>
<body>

<div class="image-container">
  <img src="{image_data_uri}" class="pcb-image" id="pcbImage">
</div>

<div class="side-panel" id="sidePanel">
  <h2 id="partName">Select a component</h2>
  <p id="partDesc" class="description placeholder-text">Inferred RE data will appear here.</p>
  <a id="partLink" class="datasheet-btn" href="#" target="_blank" style="display: none;">Search Intel</a>
</div>

<script>
  const llmResults = {json.dumps(analyzed_components)};

  const CLASS_COLORS = {{
    processor:   {{ border: '#00ff88', tag: '#005533', tagText: '#00ff88' }},
    memory:      {{ border: '#00ccff', tag: '#004455', tagText: '#00ccff' }},
    flash:       {{ border: '#ffaa00', tag: '#553300', tagText: '#ffaa00' }},
    debug_header:{{ border: '#ff4444', tag: '#551111', tagText: '#ff4444' }},
    transceiver: {{ border: '#cc44ff', tag: '#331144', tagText: '#cc44ff' }},
    regulator:   {{ border: '#888888', tag: '#333', tagText: '#888' }},
    pmic:        {{ border: '#888888', tag: '#333', tagText: '#888' }},
    opamp:       {{ border: '#888888', tag: '#333', tagText: '#888' }},
    logic:       {{ border: '#888888', tag: '#333', tagText: '#888' }},
    passive:     {{ border: '#666666', tag: '#2a2a2a', tagText: '#666' }},
    unknown:     {{ border: '#aaaaaa', tag: '#3a3a3a', tagText: '#aaa' }}
  }};

  const container = document.querySelector('.image-container');
  const img = document.getElementById('pcbImage');
  const partName = document.getElementById('partName');
  const partDesc = document.getElementById('partDesc');
  const partLink = document.getElementById('partLink');

  img.onload = () => {{
    const imgWidth = {img_width};
    const imgHeight = {img_height};

    llmResults.forEach(part => {{
      if (!part.global_box) return;

      const xmin = (part.global_box[0] / imgWidth) * 100;
      const ymin = (part.global_box[1] / imgHeight) * 100;
      const xmax = (part.global_box[2] / imgWidth) * 100;
      const ymax = (part.global_box[3] / imgHeight) * 100;

      const cls = part.component_class || 'unknown';
      const color = CLASS_COLORS[cls] || CLASS_COLORS.unknown;

      const div = document.createElement('div');
      div.className = 'hotspot';
      div.style.left = `${{xmin}}%`;
      div.style.top = `${{ymin}}%`;
      div.style.width = `${{xmax - xmin}}%`;
      div.style.height = `${{ymax - ymin}}%`;
      div.style.borderColor = color.border;

      const area = (xmax - xmin) * (ymax - ymin);
      div.style.zIndex = Math.round(10000 - area);

      div.onmouseenter = () => {{
        partName.innerText = part.part_name || "Unknown Component";

        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.innerText = cls.replace(/_/g, ' ');
        tag.style.background = color.tag;
        tag.style.color = color.tagText;

        const descP = document.createElement('p');
        descP.className = 'description';
        descP.innerText = part.description || "No analysis available.";
        descP.classList.remove('placeholder-text');

        partDesc.innerHTML = '';
        partDesc.appendChild(descP);
        partDesc.appendChild(tag);

        if (part.ocr_text) {{
          const ocrDiv = document.createElement('div');
          ocrDiv.className = 'ocr-text';
          ocrDiv.innerText = 'OCR: ' + part.ocr_text;
          partDesc.appendChild(ocrDiv);
        }}

        if (part.datasheet_link) {{
          partLink.href = part.datasheet_link;
          partLink.style.display = 'inline-block';
        }} else {{
          partLink.style.display = 'none';
        }}
      }};

      container.appendChild(div);
    }});
  }};
</script>

</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[*] Done! Open {output_html} in your web browser.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pcb.py <path_to_image.jpg>")
        sys.exit(1)

    generate_interactive_pcb(sys.argv[1])
