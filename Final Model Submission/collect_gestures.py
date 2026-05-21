"""
Gesture Data Collection Script
--------------------------------
Run from terminal: python collect_gestures.py

Controls:
  SPACE       — capture one frame
  H           — hold to auto-capture (5 frames/sec)
  N           — skip to next gesture early
  R           — redo current gesture (deletes captured frames and restarts)
  Q           — quit early

Images are saved to ./my_gesture_data/{gesture_name}/frame_XXXX.png
as 128x128 grayscale PNGs — the same format the model expects.

For best results:
  - Use a plain background (wall, desk surface)
  - Keep consistent lighting
  - Vary your hand angle and distance slightly between captures
  - Aim for the green box to contain only your hand
"""

import cv2
from pathlib import Path

# ── Define your gesture classes here ─────────────────────────────────────────
# Use simple lowercase names — these become the class labels the model predicts.
GESTURE_CLASSES = [
    'palm',       # open flat hand
    'fist',       # closed fist
    'index',      # one finger pointing up
    'thumb',      # thumbs up
    'ok',         # ok / circle sign
    'l',          # L-shape (index + thumb out)
    'c',          # curved C shape
    'down',       # pointing down
    'peace',      # peace sign or number 2
    'three',      # the number 3 on your hand
    'four',       # the number 4 on your hand
    'cross',      # crossing your index and middle finger
    'rock',       # index and pinky up with or without thumb out
    'thumbs_down',  # thumbs down
    'back_hand',    # the back of your hand
    'hand-heart',   # thumb and index finger put together to make a heart
]

IMAGES_PER_CLASS = 300   # how many frames to collect per gesture
SAVE_DIR         = Path('./my_gesture_data')
IMG_SIZE         = 128   # must match the model's input size
AUTO_CAPTURE_FPS = 5     # frames per second when holding H

# ─────────────────────────────────────────────────────────────────────────────

def count_existing(folder: Path) -> int:
    return len(list(folder.glob('*.png')))


def delete_class_images(folder: Path):
    for f in folder.glob('*.png'):
        f.unlink()


def save_frame(raw_frame, coords, folder: Path, count: int) -> bool:
    """Crop, convert to grayscale, resize, and save one frame. Returns True on success."""
    x1, y1, x2, y2 = coords
    crop    = raw_frame[y1:y2, x1:x2]
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    filename = folder / f'frame_{count:04d}.png'
    ok = cv2.imwrite(str(filename), resized)
    if not ok:
        print(f'\nWARNING: Failed to save {filename} — check disk space and permissions.')
    return ok


def draw_ui(frame, gesture_name, count, target, auto_mode):
    h, w = frame.shape[:2]

    box_size = int(min(h, w) * 0.6)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    color = (0, 200, 255) if auto_mode else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Progress bar along the bottom of the box
    progress = min(count / target, 1.0)
    bar_w    = int((x2 - x1) * progress)
    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 12), color, -1)
    cv2.rectangle(frame, (x1, y2 + 4), (x2,          y2 + 12), color, 1)

    # Gesture name + count
    cv2.putText(frame, f'Gesture: {gesture_name}', (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f'{count} / {target}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    mode_text = 'AUTO-CAPTURE (H)' if auto_mode else 'SPACE=capture  H=auto  N=next  R=redo  Q=quit'
    cv2.putText(frame, mode_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame, (x1, y1, x2, y2)


def collect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open webcam.')
        return

    # Set reasonable capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f'\nCollecting {IMAGES_PER_CLASS} images for each of {len(GESTURE_CLASSES)} gestures.')
    print(f'Saving to: {SAVE_DIR.resolve()}\n')

    for gesture_idx, gesture_name in enumerate(GESTURE_CLASSES):
        save_folder = SAVE_DIR / gesture_name
        save_folder.mkdir(parents=True, exist_ok=True)

        existing = count_existing(save_folder)
        if existing >= IMAGES_PER_CLASS:
            print(f'[{gesture_idx+1}/{len(GESTURE_CLASSES)}] {gesture_name} — already complete ({existing} images), skipping.')
            continue

        count = existing
        print(f'\n[{gesture_idx+1}/{len(GESTURE_CLASSES)}] Ready for gesture: "{gesture_name}"')
        print(f'  Starting from {count} existing images. Need {IMAGES_PER_CLASS - count} more.')
        print(f'  Hold your hand in the green box, then press SPACE to start capturing.')

        auto_mode      = False
        auto_ticker    = 0
        auto_interval  = max(1, int(30 / AUTO_CAPTURE_FPS))  # frames between auto-captures

        while count < IMAGES_PER_CLASS:
            ret, raw_frame = cap.read()
            if not ret:
                print('Failed to grab frame.')
                break

            # Draw UI annotations on a copy so the raw frame stays clean for saving
            display_frame = raw_frame.copy()
            display_frame, coords = draw_ui(
                display_frame, gesture_name, count, IMAGES_PER_CLASS, auto_mode
            )
            cv2.imshow('Gesture Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Auto-capture tick
            if auto_mode:
                auto_ticker += 1
                if auto_ticker >= auto_interval:
                    auto_ticker = 0
                    if save_frame(raw_frame, coords, save_folder, count):
                        count += 1

            if key == ord('h') or key == ord('H'):
                auto_mode   = True
                auto_ticker = 0
            elif key == ord(' '):
                auto_mode = False
                if save_frame(raw_frame, coords, save_folder, count):
                    count += 1
                    print(f'  Captured {count}/{IMAGES_PER_CLASS}', end='\r')
            elif key == ord('n') or key == ord('N'):
                print(f'\n  Skipping to next gesture with {count} images saved.')
                break
            elif key == ord('r') or key == ord('R'):
                print(f'\n  Redoing "{gesture_name}" — deleting {count} captured images.')
                delete_class_images(save_folder)
                count     = 0
                auto_mode = False
            elif key == ord('q') or key == ord('Q'):
                print('\nQuit early.')
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f'\n  Done: {gesture_name} ({count_existing(save_folder)} images saved)')

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print('\n' + '=' * 50)
    print('Collection complete. Summary:')
    total = 0
    for gesture_name in GESTURE_CLASSES:
        n = count_existing(SAVE_DIR / gesture_name)
        total += n
        status = 'OK' if n >= IMAGES_PER_CLASS else f'INCOMPLETE ({n}/{IMAGES_PER_CLASS})'
        print(f'  {gesture_name:<12} {status}')
    print(f'\nTotal images: {total}')
    print(f'Dataset path: {SAVE_DIR.resolve()}')
    print('\nNext step: open gesture_final.ipynb and set USE_CUSTOM_DATASET = True in cell 2.')


if __name__ == '__main__':
    collect()
