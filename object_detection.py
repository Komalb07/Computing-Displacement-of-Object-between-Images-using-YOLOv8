from ultralytics import YOLO
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc
from PIL import Image

plt.style.use('ggplot')

#Compute angle between two points
def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2-x1
    dy = y2-y1

    angle = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle)

    return angle_degrees

#Construct Homogeneous Transformation Matrix
def construct_homogeneous_matrix(angle, dx, dy):
    angle_rad = np.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    homogeneous_matrix = np.array([[cos_theta, -sin_theta, dx], # [ R11  R12  Tx ]
                                   [sin_theta, cos_theta, dy],  # [ R21  R22  Ty ]
                                   [0, 0, 1]])                  # [  0    0    1 ]

    return homogeneous_matrix

def update(frame):
    global current_rotation_angle_deg

    current_rotation_angle_deg += angle_increment

    if current_rotation_angle_deg > rotation_angle:
        current_rotation_angle_deg = rotation_angle

    current_rotation_angle_rad = np.radians(current_rotation_angle_deg)

    rotation_matrix = np.array([[np.cos(current_rotation_angle_rad), -np.sin(current_rotation_angle_rad)],
                                [np.sin(current_rotation_angle_rad), np.cos(current_rotation_angle_rad)]])

    new_axes = np.dot(rotation_matrix, original_axes)

    ax.clear()
    ax.imshow(image, extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.5)
    ax.quiver(0, 0, original_axes[0, 0], original_axes[1, 0], angles='xy', scale_units='xy', scale=1, color='green',
              label='Original Axes')
    ax.quiver(0, 0, original_axes[0, 1], original_axes[1, 1], angles='xy', scale_units='xy', scale=1, color='green')
    ax.quiver(0, 0, new_axes[0, 0], new_axes[1, 0], angles='xy', scale_units='xy', scale=1, color='blue',
              label='New Axes')
    ax.quiver(0, 0, new_axes[0, 1], new_axes[1, 1], angles='xy', scale_units='xy', scale=1, color='blue')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    angle_value = current_rotation_angle_deg

    angle_x = 0.2 * np.cos(np.radians(angle_value / 2))
    angle_y = 0.2 * np.sin(np.radians(angle_value / 2))

    arc = Arc([0, 0], 0.4, 0.4, 0, 0, angle_value, color='black', linewidth=1.5)
    ax.add_patch(arc)
    ax.text(angle_x, angle_y, f"{angle_value:.2f}°", fontsize=12)

    if current_rotation_angle_deg >= rotation_angle:
        final_angle_text = f"Pose: {rotation_angle:.2f} degrees"
        ax.text(-1, 1.2, final_angle_text, fontsize=12)

    ax.legend()

    if current_rotation_angle_deg >= rotation_angle:
        anim.event_source.stop()

#Import trained model into the environment
model = YOLO('/Users/komalb/PycharmProjects/pythonProject2/runs/detect/train8/weights/best.pt')

image1_path = '/Users/komalb/Downloads/cups/IMG_5773 2.jpg'
image2_path = '/Users/komalb/Downloads/cups/IMG_5776 2.jpg'

image = Image.open(image1_path)

#Train the model
#model.train(data='/Users/komalb/PycharmProjects/pythonProject2/mugs.yaml',epochs=100)

result1 = model.predict(image1_path)
result2 = model.predict(image2_path)

box1 = result1[0].boxes[0].xyxy[0].numpy()
box2 = result2[0].boxes[0].xyxy[0].numpy()

box1_centroid = np.array([(box1[0]+box1[2])/2, (box1[1]+box1[3])/2])
box2_centroid = np.array([(box2[0]+box2[2])/2, (box2[1]+box2[3])/2])

diff_in_x = round(abs(box1_centroid[0] - box2_centroid[0]), 2)
diff_in_y = round(abs(box1_centroid[1] - box2_centroid[1]), 2)
rotation_angle = round(calculate_angle(box1_centroid, box2_centroid), 2)

homogeneous_matrix = construct_homogeneous_matrix(rotation_angle, diff_in_x, diff_in_y)

num_frames = 25
angle_increment = rotation_angle / num_frames
original_axes = np.array([[1, 0], [0, 1]])
fig, ax = plt.subplots()
current_rotation_angle_deg = 0

anim = FuncAnimation(fig, update, frames=num_frames, interval=50)
print("Homogeneous Tranformation matrix is given as ")
print(homogeneous_matrix)


plt.legend()
plt.show()