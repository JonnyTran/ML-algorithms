import numpy as np
import matplotlib.pyplot as plt

from sparse_filtering import SparseFiltering

from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_extraction.image import extract_patches_2d

np.random.seed(0)

# Fetch faces
dataset = fetch_olivetti_faces(shuffle=True)
faces = dataset.data

print "faces.shape", faces.shape

n_samples, _ = faces.shape

faces_centered = faces - faces.mean(axis=0)  # global centering

faces_centered -= \
    faces_centered.mean(axis=1).reshape(n_samples, -1)  # local centering

faces_centered = \
    faces_centered.reshape(n_samples, 64, 64)  # Reshaping to 64*64 pixel images

plt.imshow(faces_centered[0], cmap=plt.get_cmap('gray'))
plt.show()
# _ = plt.title("One example from dataset with n=%s example" % n_samples)


# Extract 25 16x16 patches randomly from each image
patch_width = 16
patches = [extract_patches_2d(faces_centered[i], (patch_width, patch_width),
                              max_patches=25, random_state=i)
           for i in range(n_samples)]
patches = np.array(patches).reshape(-1, patch_width * patch_width)

# Show 25 exemplary patches
plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(patches[i].reshape(patch_width, patch_width), cmap=plt.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])
plt.show()
# _ = plt.suptitle("25 exemplary extracted patches")


n_features = 64  # How many features are learned
estimator = SparseFiltering(n_features=n_features,
                            maxfun=200,  # The maximal number of evaluations of the objective function
                            iprint=10)  # after how many function evaluations is information printed
# by L-BFGS. -1 for no information
features = estimator.fit_transform(patches)
print "features.shape", features.shape
print "estimator.w_.shape", estimator.w_.shape

plt.figure(figsize=(12, 10))
for i in range(estimator.w_.shape[0]):
    plt.subplot(int(np.sqrt(n_features)), int(np.sqrt(n_features)), i + 1)
    plt.pcolor(estimator.w_[i].reshape(patch_width, patch_width),
               cmap=plt.cm.RdYlGn, vmin=estimator.w_.min(),
               vmax=estimator.w_.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("Feature %4d" % i)
plt.tight_layout()
plt.show()


#
# plt.hist(features.flat, bins=50)
# plt.xlabel("Activation")
# plt.ylabel("Count")
# # _ = plt.title("Feature activation histogram")
