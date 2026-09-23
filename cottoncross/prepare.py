"""Extract PDF embedded image bytes, crop device edges, tile without split leakage."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from pypdf import PdfReader
from PIL import Image, ImageDraw
from .common import read_image, write_image, save_json, load_json, starts


def automatic_roi(im):
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row = np.median(g[:, w//8:7*w//8], axis=1)
    row = cv2.GaussianBlur(row[:, None], (1, 0), 9).ravel()
    mid = float(np.median(row[h//3:2*h//3]))
    top_zone = row[:int(h*.28)]
    dark = np.where(top_zone < mid * .72)[0]
    top = int(dark[-1]+24) if len(dark) else 16
    # The lower device edge is a sharp dark band; do not crop diffuse glare.
    candidates = np.where(row[int(h*.75):] < mid*.57)[0]
    bottom = int(h*.75)+int(candidates[0])-16 if len(candidates) else h-16
    # A slanted device edge can be missed by the row median. Use per-column evidence.
    lower=g[int(h*.75):,16:w-16]
    has_dark=(lower<mid*.45).any(axis=0)
    if has_dark.mean()>.1:
        first=np.argmax(lower[:,has_dark]<mid*.45,axis=0)+int(h*.75)
        bottom=min(bottom,int(np.quantile(first,.1))-24)
    return [16, max(16, top), w-16, min(h-16, bottom)]


def patch_score(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    r = g-cv2.GaussianBlur(g, (0, 0), 4)
    return float(np.mean(r > max(.025, 2.5*float(r.std()))))


def contact_sheet(paths, out, labels=None, columns=8, thumb=(160, 180)):
    n = len(paths)
    canvas = Image.new("RGB", (columns*thumb[0], ((n+columns-1)//columns)*(thumb[1]+24)), "#eef1f5")
    d = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        im = Image.open(path).convert("RGB")
        im.thumbnail(thumb)
        x, y = (i%columns)*thumb[0], (i//columns)*(thumb[1]+24)
        canvas.paste(im, (x, y))
        d.text((x+3,y+thumb[1]+3), labels[i] if labels else Path(path).stem, fill="#172337")
    canvas.save(out)


def prepare(pdf, out, config):
    out = Path(out)
    cfg = load_json(config)
    for folder in ["raw", "roi", "tiles", "previews"]:
        (out/folder).mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf)
    records, tiles, hashes = [], [], {}
    for page_idx, page in enumerate(reader.pages, 1):
        images = list(page.images)
        if not images:
            raise ValueError(f"Page {page_idx} has no embedded image; render fallback must be reviewed.")
        groups = [g for g in cfg['groups'] if g['first'] <= page_idx <= g['last']]
        if len(groups) != 1:
            raise ValueError(f"Page {page_idx} needs exactly one acquisition group.")
        group = groups[0]
        for image_idx, image in enumerate(images, 1):
            stem = f"p{page_idx:03d}_i{image_idx:02d}"
            raw = out/'raw'/f"{stem}{Path(image.name).suffix}"
            raw.write_bytes(image.data)  # Preserve the PDF image stream, no JPEG re-encoding.
            im = read_image(raw)
            sha = hashlib.sha256(image.data).hexdigest()
            if sha in hashes and hashes[sha]['split'] != group['split']:
                raise ValueError("Exact duplicate spans splits. Change groups before training.")
            roi = cfg.get('roi_overrides', {}).get(str(page_idx), automatic_roi(im))
            x0,y0,x1,y1 = roi
            if not (0 <= x0 < x1 <= im.shape[1] and 0 <= y0 < y1 <= im.shape[0]):
                raise ValueError(f"Invalid ROI {stem}: {roi}")
            crop = im[y0:y1,x0:x1]
            crop_path = out/'roi'/f"{stem}.png"
            write_image(crop_path,crop)
            rec = dict(id=stem,page=page_idx,image_index=image_idx,source_name=Path(pdf).name,
                       raw=f"raw/{raw.name}",roi=f"roi/{crop_path.name}",roi_xyxy=roi,
                       width=im.shape[1],height=im.shape[0],sha256=sha,
                       group=group['id'],split=group['split'],duplicate_of=hashes.get(sha,{}).get('id'))
            hashes[sha] = rec
            records.append(rec)
            preview = im.copy()
            cv2.rectangle(preview,(x0,y0),(x1-1,y1-1),(40,220,40),8)
            preview=cv2.resize(preview,(384,512))
            write_image(out/'previews'/f"{stem}.jpg",preview)
            size = cfg['tile_size']
            for ty in starts(crop.shape[0],size,cfg['stride']):
                for tx in starts(crop.shape[1],size,cfg['stride']):
                    tile=crop[ty:ty+size,tx:tx+size]
                    tid=f"{stem}_y{ty:04d}_x{tx:04d}"
                    write_image(out/'tiles'/f"{tid}.png",tile)
                    tiles.append(dict(id=tid,source_id=stem,page=page_idx,group=group['id'],split=group['split'],
                                      path=f"tiles/{tid}.png",xyxy_original=[x0+tx,y0+ty,x0+tx+tile.shape[1],y0+ty+tile.shape[0]],
                                      signal_score=patch_score(tile)))
        print(f"extracted page {page_idx}/{len(reader.pages)}",flush=True)
    save_json(out/'manifest.json',records)
    save_json(out/'tiles.json',tiles)
    save_json(out/'preparation_config.json',cfg)
    contact_sheet([out/'previews'/f"{r['id']}.jpg" for r in records],out/'crop_contact.jpg',
                  [f"{r['page']:02d} {r['split']}" for r in records])
    # Exact duplicate audit plus nearest-neighbor fingerprints, for human review only.
    descriptors=[]
    for r in records:
        x=read_image(out/r['roi'],gray=True)
        x=cv2.resize(x,(48,64)).astype(np.float32)
        x=(x-x.mean())/(x.std()+1e-6)
        descriptors.append(x.ravel())
    d=np.array(descriptors)
    distances=np.mean((d[:,None]-d[None,:])**2,axis=-1)
    np.fill_diagonal(distances,np.inf)
    audit=[dict(id=r['id'],nearest=records[int(np.argmin(distances[i]))]['id'],
                distance=float(distances[i].min()),warning="background-sensitive; not proof of specimen identity")
           for i,r in enumerate(records)]
    save_json(out/'similarity_audit.json',audit)
    summary=dict(pdf_pages=len(reader.pages),extracted_images=len(records),exact_unique=len(hashes),
                 roi_count=len(records),tile_count=len(tiles),
                 split_images={s:sum(r['split']==s for r in records) for s in ['train','val','test']},
                 split_tiles={s:sum(r['split']==s for r in tiles) for s in ['train','val','test']},
                 specimen_independence_verified=False)
    save_json(out/'summary.json',summary)
    print(summary)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pdf',required=True)
    p.add_argument('--out',default='data/real')
    p.add_argument('--config',default='configs/dataset.json')
    a=p.parse_args(); prepare(a.pdf,a.out,a.config)
