import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Flatten image into 1D array
pixels = image.flatten().astype(np.float64)

# Number of Gaussians
K = 2

# Initialize parameters
np.random.seed(0)

means = np.random.choice(pixels, K)
variances = np.full(K, 500.0)
weights = np.full(K, 1 / K)

# Gaussian PDF
def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * \
           np.exp(-(x - mean) ** 2 / (2 * var))

# EM Algorithm
max_iterations = 50

for i in range(max_iterations):

    prev_means = means.copy()

    # E-step
    responsibilities = np.zeros((len(pixels), K))

    for k in range(K):
        responsibilities[:, k] = (
            weights[k] *
            gaussian(pixels, means[k], variances[k])
        )

    sums = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= (sums + 1e-10)

    # M-step
    Nk = responsibilities.sum(axis=0)

    for k in range(K):

        means[k] = np.sum(
            responsibilities[:, k] * pixels
        ) / Nk[k]

        variances[k] = np.sum(
            responsibilities[:, k] *
            (pixels - means[k]) ** 2
        ) / Nk[k]

        variances[k] += 1e-5

        weights[k] = Nk[k] / len(pixels)

    # Convergence check
    if np.allclose(means, prev_means, atol=1e-3):
        print(f"Converged at iteration {i}")
        break
# Assign each pixel to best Gaussian
labels = np.argmax(responsibilities, axis=1)

# Convert labels back to image
segmented = (labels * (255 // (K - 1))).reshape(image.shape).astype(np.uint8)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Segmented")
plt.imshow(segmented, cmap='gray')

plt.show()