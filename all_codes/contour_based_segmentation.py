import cv2
import numpy as np

# Load the original image and the segmented mask
original_image = cv2.imread('/home/labhansh/Open3D/MiDaS/input/cat_og.jpg')  # Load your original image
segmented_image = cv2.imread('/home/labhansh/Open3D/MiDaS/input/cat_panoptic_resized.png')  # Load your segmented image

# Ensure the images are loaded successfully
if segmented_image is not None and original_image is not None:
    # Convert the segmented image to grayscale
    segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the segmented images
    import os
    output_directory = '/home/labhansh/Open3D/MiDaS/segmented_images_yo'
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over each contour and save as a separate image
    for i, contour in enumerate(contours):
        # Create a mask for the current segment
        mask = np.zeros_like(segmented_gray)
        cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)

        # Extract the segment from the original image
        segmented_segment = cv2.bitwise_and(original_image, original_image, mask=mask)

        # Save the segment as a separate image
        segment_filename = os.path.join(output_directory, f'segment_{i}.jpg')
        cv2.imwrite(segment_filename, segmented_segment)

        print(f"Saved segment {i} as {segment_filename}")

else:
    print("Images not found or could not be loaded.")




'''
# Convert the segmented mask to a binary mask with distinct values for each segment
unique_values = np.unique(segmented_mask)
segmented_objects = []
print(unique_values)

for val in unique_values:
    if val == 0:
        continue  # Ignore the background
    mask = np.where(segmented_mask == val, 1, 0)
    segmented_objects.append(mask)

# Apply GrabCut to each segment and save as separate images
for idx, segment_mask in enumerate(segmented_objects):
    # Create a GrabCut mask
    grabcut_mask = np.zeros(segment_mask.shape, dtype=np.uint8)

    # Define the foreground and background models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle based on the segment_mask (you may need to customize this)
    contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Apply GrabCut algorithm
    cv2.grabCut(original_image, grabcut_mask, (x, y, w, h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Extract the segmented object based on GrabCut mask
    segmented_object = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 255).astype('uint8')

    # Save the segmented object as a separate image
    cv2.imwrite(f'/home/labhansh/Open3D/MiDaS/input/segmented_object_{idx}.jpg', segmented_object)

# Optionally, display the segmented objects
for idx, segment_mask in enumerate(segmented_objects):
    cv2.imshow(f'Segmented Object {idx}', segmented_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''