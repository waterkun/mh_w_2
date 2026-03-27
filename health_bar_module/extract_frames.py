"""Extract frames from gameplay video and crop health bar ROI.

Usage:
    python extract_frames.py --video path/to/video.mp4 [--interval 10]
    python extract_frames.py --recrop                   # re-crop ROIs from existing frames
"""

import argparse
import os

import cv2


DEFAULT_ROI = (146, 71, 728, 31)  # x, y, w, h


def extract(video_path, output_dir, interval=10, roi=DEFAULT_ROI):
    os.makedirs(os.path.join(output_dir, "rois"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames, {fps:.1f} FPS")
    print(f"Extracting every {interval} frames -> ~{total // interval} images")

    x, y, w, h = roi
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            name = f"frame_{count:05d}.png"
            # Save ROI crop
            roi_img = frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(output_dir, "rois", name), roi_img)
            # Save full frame for reference
            cv2.imwrite(os.path.join(output_dir, "frames", name), frame)
            count += 1

            if count % 100 == 0:
                print(f"  {count} frames extracted...")

        frame_idx += 1

    cap.release()
    print(f"Done: {count} frames saved to {output_dir}")


def recrop(output_dir, roi=DEFAULT_ROI):
    """Re-crop ROIs from existing full frames with a new ROI."""
    frames_dir = os.path.join(output_dir, "frames")
    rois_dir = os.path.join(output_dir, "rois")
    os.makedirs(rois_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg")))
    if not files:
        print(f"No frames found in {frames_dir}")
        return

    x, y, w, h = roi
    print(f"Re-cropping {len(files)} frames with ROI ({x}, {y}, {w}, {h})")

    for i, fname in enumerate(files):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            continue
        roi_img = frame[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(rois_dir, fname), roi_img)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1} / {len(files)}")

    print(f"Done: {len(files)} ROIs saved to {rois_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract health bar ROI from video")
    parser.add_argument("--video", default=None, help="Path to gameplay video")
    parser.add_argument("--interval", type=int, default=10,
                        help="Extract every N frames (default: 10)")
    parser.add_argument("--roi", default="146,71,728,31",
                        help="ROI as x,y,w,h (default: 146,71,728,31)")
    parser.add_argument("--output", default="health_bar_data",
                        help="Output directory (default: health_bar_data)")
    parser.add_argument("--recrop", action="store_true",
                        help="Re-crop ROIs from existing frames (no video needed)")
    args = parser.parse_args()

    roi = tuple(int(v) for v in args.roi.split(","))

    if args.recrop:
        recrop(args.output, roi)
    elif args.video:
        extract(args.video, args.output, args.interval, roi)
    else:
        parser.error("Either --video or --recrop is required")


if __name__ == "__main__":
    main()
