import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------

IMG1_PATH = Path("data/assignment_2/first_image.jpg")
IMG2_PATH = Path("data/assignment_2/second_image.jpg")

# -----------------------------------------------------------
# Load images
# -----------------------------------------------------------

img1 = cv2.imread(str(IMG1_PATH), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(IMG2_PATH), cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not load images")

# -----------------------------------------------------------
# EX 4.2 — SIFT keypoints + descriptors
# -----------------------------------------------------------

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print("Image1 keypoints:", len(kp1))
print("Image2 keypoints:", len(kp2))

# -----------------------------------------------------------
# Visualize keypoints
# -----------------------------------------------------------

img1_kp = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

img2_kp = cv2.drawKeypoints(
    img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Keypoints Image 1")
plt.imshow(img1_kp, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Keypoints Image 2")
plt.imshow(img2_kp, cmap="gray")

plt.show()

# -----------------------------------------------------------
# EX 4.4 — Feature Matching
# -----------------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Total matches:", len(matches))
print("Good matches:", len(good_matches))

# -----------------------------------------------------------
# Draw matches
# -----------------------------------------------------------

matched_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

plt.figure(figsize=(12, 6))
plt.title("SIFT Feature Matches")
plt.imshow(matched_img)
plt.axis("off")
plt.show()

import numpy as np
import cv2

# -----------------------------------------------------------
# 8 Point Algorithm
# -----------------------------------------------------------


def compute_fundamental_matrix(pts1, pts2):

    n = pts1.shape[0]

    A = []

    for i in range(n):
        x, y = pts1[i]
        xp, yp = pts2[i]

        A.append([x * xp, x * yp, x, y * xp, y * yp, y, xp, yp, 1])

    A = np.array(A)

    # Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)

    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    return F / F[2, 2]


# -----------------------------------------------------------
# Manual point selection
# -----------------------------------------------------------

manual_pts1 = []
manual_pts2 = []


def click_img1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_pts1.append([x, y])
        print("img1:", x, y)


def click_img2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        manual_pts2.append([x, y])
        print("img2:", x, y)


cv2.imshow("Image1", img1)
cv2.setMouseCallback("Image1", click_img1)

cv2.imshow("Image2", img2)
cv2.setMouseCallback("Image2", click_img2)

print("Click 8 corresponding points in each image, then press any key")

cv2.waitKey(0)
cv2.destroyAllWindows()

manual_pts1 = np.array(manual_pts1[:8])
manual_pts2 = np.array(manual_pts2[:8])

F_manual = compute_fundamental_matrix(manual_pts1, manual_pts2)

print("\nFundamental Matrix (Manual Points):")
print(F_manual)

# -----------------------------------------------------------
# Fundamental matrix from feature matches
# -----------------------------------------------------------

pts1 = []
pts2 = []

for m in good_matches[:50]:  # take first 50 matches
    pts1.append(kp1[m.queryIdx].pt)
    pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

F_auto = compute_fundamental_matrix(pts1, pts2)

print("\nFundamental Matrix (Feature Matches):")
print(F_auto)
