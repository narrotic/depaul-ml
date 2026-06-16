import cv2
import numpy as np
import json

width, height = 900, 650
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

rectangles = []

drawing = False
dragging = False

start_point = None
selected_rect = None
selected_corner = None

mode = "translation"
corner_radius = 10


def redraw():
    canvas[:] = 255

    for rect in rectangles:
        pts = rect.astype(int)
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)

        for p in pts:
            cv2.circle(canvas, tuple(p), 5, (0, 0, 255), -1)

    cv2.imshow("Canvas", canvas)


def find_corner(x, y):
    global selected_rect, selected_corner

    for rect in rectangles:
        for i, (cx, cy) in enumerate(rect):
            if abs(cx - x) < corner_radius and abs(cy - y) < corner_radius:
                selected_rect = rect
                selected_corner = i
                return True
    return False


def transform(rect, idx, x, y):

    new_pt = np.array([x, y], dtype=np.float32)

    if mode == "translation":
        delta = new_pt - rect[idx]
        rect += delta

    elif mode == "rigid":

        center = rect.mean(axis=0)

        old = rect[idx] - center
        new = new_pt - center

        theta = np.arctan2(new[1], new[0]) - np.arctan2(old[1], old[0])

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        rect[:] = (rect - center) @ R.T + center

        delta = new_pt - rect[idx]
        rect += delta

    elif mode == "similarity":

        center = rect.mean(axis=0)

        old = rect[idx] - center
        new = new_pt - center

        scale = np.linalg.norm(new) / np.linalg.norm(old)

        theta = np.arctan2(new[1], new[0]) - np.arctan2(old[1], old[0])

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        rect[:] = ((rect - center) * scale) @ R.T + center

        delta = new_pt - rect[idx]
        rect += delta

    elif mode == "affine":

        src = rect[:3].astype(np.float32)
        dst = src.copy()

        dst[idx % 3] = new_pt

        M = cv2.getAffineTransform(src, dst)

        pts = np.hstack([rect, np.ones((4, 1))])
        rect[:] = (M @ pts.T).T

    elif mode == "perspective":

        src = rect.astype(np.float32)
        dst = src.copy()

        dst[idx] = new_pt

        H = cv2.getPerspectiveTransform(src, dst)

        pts = np.hstack([rect, np.ones((4, 1))]).T

        t = H @ pts
        t /= t[2]

        rect[:] = t[:2].T


def mouse(event, x, y, flags, param):

    global drawing, start_point, dragging

    if event == cv2.EVENT_LBUTTONDOWN:

        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            drawing = True
            start_point = (x, y)

        else:
            if find_corner(x, y):
                dragging = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            redraw()

            p1 = start_point
            p2 = (x, y)

            pts = np.array(
                [[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]]
            )

            cv2.polylines(canvas, [pts.astype(int)], True, (200, 0, 0), 2)
            cv2.imshow("Canvas", canvas)

        elif dragging:

            transform(selected_rect, selected_corner, x, y)
            redraw()

    elif event == cv2.EVENT_LBUTTONUP:

        global rectangles

        if drawing:

            drawing = False

            p1 = start_point
            p2 = (x, y)

            rect = np.array(
                [[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]],
                dtype=np.float32,
            )

            rectangles.append(rect)

            redraw()

        dragging = False


def save_rects():
    data = [r.tolist() for r in rectangles]
    json.dump(data, open("rectangles.json", "w"))
    print("saved")


def load_rects():
    global rectangles
    try:
        data = json.load(open("rectangles.json"))
        rectangles = [np.array(r, dtype=np.float32) for r in data]
        redraw()
        print("loaded")
    except:
        print("no file")


cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", mouse)

print("SHIFT+drag : create rectangle")
print("drag corner: transform")
print("t/r/s/a/p  : change mode")
print("k save | l load | q quit")

redraw()

while True:

    k = cv2.waitKey(1) & 0xFF

    if k == ord("q"):
        break

    elif k == ord("t"):
        mode = "translation"
        print("translation")

    elif k == ord("r"):
        mode = "rigid"
        print("rigid")

    elif k == ord("s"):
        mode = "similarity"
        print("similarity")

    elif k == ord("a"):
        mode = "affine"
        print("affine")

    elif k == ord("p"):
        mode = "perspective"
        print("perspective")

    elif k == ord("k"):
        save_rects()

    elif k == ord("l"):
        load_rects()

cv2.destroyAllWindows()
