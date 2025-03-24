import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(2)

selected_pts = []

def select_points(event, x, y, flags, param):
    """ Mouse callback function to store clicked points """
    global selected_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_pts) < 4:
        selected_pts.append((x, y))
        print(f"Point {len(selected_pts)}: ({x}, {y})")

cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", select_points)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Select Points", frame)

    if len(selected_pts) == 4:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Convert points to numpy array
src_pts = np.array(selected_pts, dtype=np.float32)

# Define destination points (Fix flipped issue)
width, height = 600, 400
dst_pts = np.float32([
    [0, 0],        # Top-left maps to (0,0)
    [width, 0],    # Top-right maps to (width,0)
    [0, height],   # Bottom-left maps to (0,height)
    [width, height] # Bottom-right maps to (width, height)
])

# Draw selected points on the original frame
for i, point in enumerate(selected_pts):
    cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Draw red dots
    cv2.putText(frame, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.imshow("Selected Region", frame)
cv2.waitKey(5000)  # Pause to inspect points

# Compute transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped = cv2.warpPerspective(frame, M, (width, height))

    # Crop black areas (adjust values as needed)
    cropped = warped[50:height-50, 50:width-50]  

    cv2.imshow("Bird's Eye View", cropped)
    cv2.imshow("Original Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
