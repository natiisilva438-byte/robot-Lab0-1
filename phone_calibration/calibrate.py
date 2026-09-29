"""Calibrate a pinhole camera from a 9x6-square, 30 mm checkerboard.

Example: python calibrate.py --input data/raw --supplemental data/supplemental_raw --output results
The pattern argument counts INTERNAL corners: 8 columns x 5 rows.
Images are measured at native resolution, then their corner coordinates
are mapped to the original 4096x3072 image grid. Portrait supplemental images
are rotated clockwise before detection; the rotation is recorded per image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    buffer.tofile(str(path))


def find_corners(gray: np.ndarray, pattern: tuple[int, int], scale: float) -> tuple[np.ndarray | None, str]:
    # Detection scale is selected from image resolution, independent of batch.
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    method = "classic"
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(small, pattern)
        method = "SB"
    if not ok:
        return None, "none"
    corners = (corners / scale).astype(np.float32)
    cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
    return corners, method


def calibrate(obj: np.ndarray, views: list[dict], size: tuple[int, int]):
    objects = [obj for _ in views]
    images = [v["corners"] for v in views]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, images, size, None, None)
    errors = []
    for view, rvec, tvec in zip(views, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residual = view["corners"].reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return rms, K, dist, rvecs, tvecs, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, help="Additional original photographs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=8, help="internal corner columns")
    parser.add_argument("--rows", type=int, default=5, help="internal corner rows")
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    source, out = args.input.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "corners").mkdir(exist_ok=True)
    for stale_overlay in (out / "corners").glob("*.jpg"):
        stale_overlay.unlink()
    image_files = [(p, "initial") for p in sorted(source.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not image_files:
        raise SystemExit("No image files found")
    if args.supplemental:
        image_files.extend((p, "supplemental") for p in sorted(args.supplemental.resolve().iterdir())
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    obj = np.zeros((args.columns * args.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    views, manifest = [], []
    first = read_image(image_files[0][0])
    size = (first.shape[1], first.shape[0])
    if size[0] < size[1]:
        raise ValueError("Initial reference image must be landscape")
    for index, (path, batch) in enumerate(image_files, 1):
        image = read_image(path)
        native_height, native_width = image.shape[:2]
        rotation = "none"
        if native_height > native_width:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotation = "90_cw_to_landscape"
        height, width = image.shape[:2]
        if abs(width / height - size[0] / size[1]) > 0.002:
            raise ValueError(f"Aspect ratio incompatible with canonical {size}: {path} ({width}x{height})")
        scale_x, scale_y = size[0] / width, size[1] / height
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners_native, method = find_corners(gray, (args.columns, args.rows), 0.5 if width >= 3000 else 1.0)
        corners = None if corners_native is None else corners_native * np.array([scale_x, scale_y], np.float32)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = {"id": index, "filename": path.name, "sha256": digest,
                 "source_batch": batch, "width_px": native_width, "height_px": native_height,
                 "rotation": rotation, "scale_x": scale_x, "scale_y": scale_y,
                 "detected": corners is not None, "detector": method,
                 "used": corners is not None,
                 "distance_group": ("near" if index <= 10 else "middle" if index <= 21 else "far")
                                   if batch == "initial" else "unclassified"}
        manifest.append(entry)
        if corners is not None:
            views.append({"entry": entry, "corners": corners, "path": path})
            overlay = image.copy()
            cv2.drawChessboardCorners(overlay, (args.columns, args.rows), corners_native, True)
            save_image(out / "corners" / f"{index:02d}.jpg", cv2.resize(overlay, (1024, 768)))
        print(f"{index:02d}/{len(image_files)} {batch} {path.name}: {method} ({native_width}x{native_height})", flush=True)
    if len(views) < 8:
        raise SystemExit(f"Only {len(views)} checkerboards found; at least 8 are needed")
    all_rms, _, _, _, _, all_errors = calibrate(obj, views, size)
    for view, error in zip(views, all_errors):
        view["entry"]["initial_rms_px"] = error
    gross_rejected = [v for v, err in zip(views, all_errors) if err > 3.0]
    gross_ids = {v["entry"]["id"] for v in gross_rejected}
    screened = [v for v in views if v["entry"]["id"] not in gross_ids]
    screening_rms, _, _, _, _, screening_errors = calibrate(obj, screened, size)
    for view, error in zip(screened, screening_errors):
        view["entry"]["screening_rms_px"] = error
    med = float(np.median(screening_errors))
    mad = float(np.median(np.abs(np.array(screening_errors) - med)))
    threshold = max(1.5, med + 3 * 1.4826 * mad)
    # A gross-error pass prevents one bad board from masking valid views.
    rejected = [v for v, err in zip(screened, screening_errors) if err > threshold]
    rejected_ids = gross_ids | {v["entry"]["id"] for v in rejected}
    if len(views) - len(rejected_ids) < 8:
        raise SystemExit("Outlier screening leaves fewer than eight images")
    kept = [v for v in views if v["entry"]["id"] not in rejected_ids]
    for v in gross_rejected:
        v["entry"]["used"] = False
        v["entry"]["reason"] = "gross_initial_reprojection_error"
    for v in rejected:
        v["entry"]["used"] = False
        v["entry"]["reason"] = "high_screening_reprojection_error"
    rms, K, dist, rvecs, tvecs, errors = calibrate(obj, kept, size)
    residual_records = []
    for view, rvec, tvec, err in zip(kept, rvecs, tvecs, errors):
        entry = view["entry"]
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        observed = view["corners"].reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        delta = observed - projected
        rotation, _ = cv2.Rodrigues(rvec)
        board_normal = rotation[:, 2]
        entry["rms_px"] = err
        entry["tvec_mm"] = tvec.reshape(3).tolist()
        entry["rvec_rad"] = rvec.reshape(3).tolist()
        entry["distance_mm"] = float(np.linalg.norm(tvec))
        if entry["source_batch"] == "supplemental":
            entry["distance_group"] = ("near" if entry["distance_mm"] < 1000 else
                                       "middle" if entry["distance_mm"] < 1800 else "far")
        entry["optical_depth_mm"] = float(tvec[2, 0])
        entry["tilt_deg"] = float(np.degrees(np.arccos(np.clip(abs(board_normal[2]), 0, 1))))
        edge = observed[args.columns - 1] - observed[0]
        raw_angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
        entry["in_plane_deg"] = (raw_angle + 90) % 180 - 90
        hull = cv2.convexHull(observed.astype(np.float32))
        entry["board_area_pct"] = float(100 * cv2.contourArea(hull) / (size[0] * size[1]))
        entry["center_x_pct"] = float(100 * observed[:, 0].mean() / size[0])
        entry["center_y_pct"] = float(100 * observed[:, 1].mean() / size[1])
        for n, (p, q, d) in enumerate(zip(observed, projected, delta), 1):
            residual_records.append({"image_id": entry["id"], "corner_id": n,
                                     "observed_x": p[0], "observed_y": p[1],
                                     "projected_x": q[0], "projected_y": q[1],
                                     "dx_px": d[0], "dy_px": d[1]})
    # Acquisition groups are assigned above from the image sequence and retained
    # even for excluded images. Pose estimates verify their separation.
    result = {"camera": "HONOR 200 (reported by photographer; lens and mode unverified)",
              "image_size_px": size, "checkerboard_internal_corners": [args.columns, args.rows],
              "square_size_mm": args.square_mm, "input_images": len(image_files),
              "normalization": "detect at native resolution; rotate portrait supplemental 90 CW; scale corners to canonical pixel grid",
              "detected_images": len(views),
              "used_images": len(kept), "rejected_images": sorted(rejected_ids),
              "initial_rms_px": all_rms, "screening_rms_px": screening_rms,
              "gross_error_threshold_px": 3.0, "gross_rejected_images": sorted(gross_ids),
              "final_rms_px": rms,
              "outlier_threshold_px": threshold,
              "camera_matrix": K.tolist(), "distortion_coefficients": dist.reshape(-1).tolist(),
              "opencv_version": cv2.__version__, "views": manifest}
    (out / "calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        columns = ["id", "filename", "sha256", "source_batch", "width_px", "height_px", "rotation",
                   "scale_x", "scale_y", "detected", "detector", "used",
                   "reason", "distance_group", "distance_mm", "optical_depth_mm", "tilt_deg", "in_plane_deg",
                   "board_area_pct", "center_x_pct", "center_y_pct", "initial_rms_px",
                   "screening_rms_px", "rms_px"]
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    with (out / "residuals.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(residual_records[0]))
        writer.writeheader()
        writer.writerows(residual_records)
    fs = cv2.FileStorage(str(out / "camera.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0]); fs.write("image_height", size[1])
    fs.write("camera_matrix", K); fs.write("distortion_coefficients", dist)
    fs.release()
    if args.copy_images:
        for path, batch in image_files:
            raw = out.parent / "data" / ("raw" if batch == "initial" else "supplemental_raw")
            raw.mkdir(parents=True, exist_ok=True)
            if path.resolve() != (raw / path.name).resolve():
                shutil.copy2(path, raw / path.name)
    print(json.dumps({"detected": len(views), "used": len(kept), "RMS_px": rms,
                      "K": K.tolist(), "dist": dist.reshape(-1).tolist()}, indent=2))


if __name__ == "__main__":
    main()
