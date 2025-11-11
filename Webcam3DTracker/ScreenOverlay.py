import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard
from PyQt5 import QtWidgets, QtGui, QtCore
import sys

# Screen and mouse control setup
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False
filter_length = 10
gaze_length = 350

# Heatmap settings
heatmap_buffer = deque(maxlen=100)  # Store last 100 gaze points
heatmap_decay = 0.95  # Fade rate
gaussian_size = 100  # Size of gaussian blob

# --- 3D monitor plane state (world space) ---
monitor_corners = None
monitor_center_w = None
monitor_normal_w = None
units_per_cm = None

# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets
calibration_offset_yaw = 0
calibration_offset_pitch = 0
calib_step = 0

# Buffers to store recent gaze data for smoothing
combined_gaze_directions = deque(maxlen=filter_length)

# reference matrices
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# Eye sphere tracking variables
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

# Global for screen position sharing between threads
current_screen_pos = [CENTER_X, CENTER_Y]
screen_pos_lock = threading.Lock()

# Global overlay reference
overlay_widget = None

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Nose landmark indices
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]


class TransparentHeatmapOverlay(QtWidgets.QWidget):
    """Transparent fullscreen overlay to show gaze heatmap"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # Click through
        
        # Set to fullscreen
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        # Heatmap array (accumulates gaze points)
        self.heatmap = np.zeros((MONITOR_HEIGHT, MONITOR_WIDTH), dtype=np.float32)
        
        # Timer for updating display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)  # Update at ~33 fps
        
        # Show status text
        self.show_help = True
        
    def update_display(self):
        """Update the heatmap display"""
        # Decay existing heatmap
        self.heatmap *= heatmap_decay
        
        # Add current gaze point
        with screen_pos_lock:
            x, y = current_screen_pos
        
        if 0 <= x < MONITOR_WIDTH and 0 <= y < MONITOR_HEIGHT:
            # Add gaussian blob at gaze position
            self.add_gaussian_blob(x, y, gaussian_size, intensity=50.0)
        
        self.update()
    
    def add_gaussian_blob(self, cx, cy, size, intensity=1.0):
        """Add a gaussian blob to the heatmap at position (cx, cy)"""
        half_size = size // 2
        
        # Define region to update
        x_start = max(0, cx - half_size)
        x_end = min(MONITOR_WIDTH, cx + half_size)
        y_start = max(0, cy - half_size)
        y_end = min(MONITOR_HEIGHT, cy + half_size)
        
        if x_end <= x_start or y_end <= y_start:
            return
        
        # Create gaussian kernel
        y_range = np.arange(y_start, y_end)
        x_range = np.arange(x_start, x_end)
        xx, yy = np.meshgrid(x_range, y_range)
        
        # Gaussian formula
        sigma = size / 6.0
        gaussian = intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        
        # Add to heatmap
        self.heatmap[y_start:y_end, x_start:x_end] += gaussian
    
    def paintEvent(self, event):
        """Draw the heatmap overlay"""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Convert heatmap to colored image
        heatmap_normalized = np.clip(self.heatmap, 0, 255).astype(np.uint8)
        
        # Apply colormap (hot/jet style)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        height, width = heatmap_colored.shape[:2]
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(heatmap_colored.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Draw with transparency
        painter.setOpacity(0.5)
        painter.drawImage(0, 0, q_img)
        
        # Draw current gaze point (white circle)
        painter.setOpacity(1.0)
        with screen_pos_lock:
            x, y = current_screen_pos
        
        if left_sphere_locked and right_sphere_locked:
            # Draw crosshair at gaze point
            pen = QtGui.QPen(QtCore.Qt.white, 3)
            painter.setPen(pen)
            crosshair_size = 20
            painter.drawLine(x - crosshair_size, y, x + crosshair_size, y)
            painter.drawLine(x, y - crosshair_size, x, y + crosshair_size)
            
            # Draw circle
            pen = QtGui.QPen(QtCore.Qt.green, 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPoint(x, y), 15, 15)
        
        # Draw help text
        if self.show_help:
            painter.setOpacity(0.8)
            pen = QtGui.QPen(QtCore.Qt.white, 2)
            painter.setPen(pen)
            font = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
            painter.setFont(font)
            
            help_text = [
                "Eye Tracker - Screen Overlay Mode",
                "",
                "C = Calibrate (look at screen center)",
                "R = Reset heatmap",
                "H = Toggle help",
                "Q = Quit",
                "",
                f"Status: {'CALIBRATED' if left_sphere_locked and right_sphere_locked else 'NOT CALIBRATED'}",
                f"Gaze: ({x}, {y})"
            ]
            
            y_offset = 30
            for line in help_text:
                painter.drawText(20, y_offset, line)
                y_offset += 25


def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_scale(points_3d):
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0


def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])
    
    center = np.mean(points_3d, axis=0)
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1
    
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()
    
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1
    
    return center, R_final, points_3d


def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """Convert 3D gaze direction vector to 2D screen coordinates"""
    reference_forward = np.array([0, 0, -1])
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)
    
    # Horizontal (yaw) angle
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad
    
    # Vertical (pitch) angle
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad
    
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    
    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = - yaw_deg
    
    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg
    
    yawDegrees = 5 * 3
    pitchDegrees = 2.0 * 2.5
    
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch
    
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)
    
    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))
    
    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg


def eye_tracking_thread():
    """Main eye tracking loop running in separate thread"""
    global left_sphere_locked, right_sphere_locked
    global left_sphere_local_offset, right_sphere_local_offset
    global left_calibration_nose_scale, right_calibration_nose_scale
    global calibration_offset_yaw, calibration_offset_pitch
    global w, h, overlay_widget
    
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_radius = 20
    
    print("="*60)
    print("[Eye Tracker - Screen Overlay Mode] Started")
    print("="*60)
    print("[Instructions]")
    print("  C = Calibrate (look at screen center and press C)")
    print("  R = Reset heatmap")
    print("  H = Toggle help overlay")
    print("  Q = Quit")
    print("="*60)
    
    # Small control window for keyboard input (always on top)
    control_window_name = "Eye Tracker Control - CLICK HERE TO USE KEYS"
    cv2.namedWindow(control_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window_name, 500, 120)
    cv2.setWindowProperty(control_window_name, cv2.WND_PROP_TOPMOST, 1)
    control_img = np.zeros((120, 500, 3), dtype=np.uint8)
    cv2.putText(control_img, "Eye Tracker Active", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(control_img, "CLICK THIS WINDOW, then press keys", (10, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    cv2.putText(control_img, "C=Calibrate R=Reset H=Help Q=Quit", (10, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow(control_window_name, control_img)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            left_iris_idx = 468
            right_iris_idx = 473
            left_iris = face_landmarks[left_iris_idx]
            right_iris = face_landmarks[right_iris_idx]
            
            head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
                frame,
                face_landmarks,
                nose_indices,
                R_ref_nose,
                color=(0, 255, 0),
                size=80
            )
            
            iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
            iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
            
            if left_sphere_locked and right_sphere_locked:
                current_nose_scale = compute_scale(nose_points_3d)
                scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
                scaled_offset = left_sphere_local_offset * scale_ratio
                sphere_world_l = head_center + R_final @ scaled_offset
                
                scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
                scaled_offset_r = right_sphere_local_offset * scale_ratio_r
                sphere_world_r = head_center + R_final @ scaled_offset_r
                
                # Calculate gaze directions
                left_gaze_dir = iris_3d_left - sphere_world_l
                left_gaze_dir /= np.linalg.norm(left_gaze_dir)
                
                right_gaze_dir = iris_3d_right - sphere_world_r
                right_gaze_dir /= np.linalg.norm(right_gaze_dir)
                
                # Combined gaze
                raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
                raw_combined_direction /= np.linalg.norm(raw_combined_direction)
                
                combined_gaze_directions.append(raw_combined_direction)
                avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
                avg_combined_direction /= np.linalg.norm(avg_combined_direction)
                
                # Convert to screen coordinates
                screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                    avg_combined_direction, 
                    calibration_offset_yaw, 
                    calibration_offset_pitch
                )
                
                # Update global screen position
                with screen_pos_lock:
                    current_screen_pos[0] = screen_x
                    current_screen_pos[1] = screen_y
        
        # Update control window status
        control_img = np.zeros((120, 500, 3), dtype=np.uint8)
        if left_sphere_locked and right_sphere_locked:
            cv2.putText(control_img, "STATUS: CALIBRATED", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(control_img, "STATUS: NOT CALIBRATED", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(control_img, "CLICK THIS WINDOW, then press keys", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(control_img, "C=Calibrate R=Reset H=Help Q=Quit", (10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(control_window_name, control_img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Eye Tracker] Shutting down...")
            break
        elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
            # Calibration
            print("[Calibration] Calibrating... Look at the CENTER of your screen!")
            current_nose_scale = compute_scale(nose_points_3d)
            
            # Lock LEFT eye
            left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
            camera_dir_world = np.array([0, 0, 1])
            camera_dir_local = R_final.T @ camera_dir_world
            left_sphere_local_offset += base_radius * camera_dir_local
            left_calibration_nose_scale = current_nose_scale
            left_sphere_locked = True
            
            # Lock RIGHT eye
            right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
            right_sphere_local_offset += base_radius * camera_dir_local
            right_calibration_nose_scale = current_nose_scale
            right_sphere_locked = True
            
            print("[Calibration] ✓ Eye tracking calibrated! Move your eyes around.")
            print("[Calibration] Green crosshair shows where you're looking.")
        elif key == ord('r'):
            # Reset heatmap
            if overlay_widget is not None:
                overlay_widget.heatmap.fill(0)
                print("[Heatmap] Reset! Starting fresh.")
        elif key == ord('h'):
            # Toggle help
            if overlay_widget is not None:
                overlay_widget.show_help = not overlay_widget.show_help
                print(f"[Help] {'Shown' if overlay_widget.show_help else 'Hidden'}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main application"""
    global overlay_widget
    
    # Start eye tracking in separate thread
    tracking_thread = threading.Thread(target=eye_tracking_thread, daemon=True)
    tracking_thread.start()
    
    # Start Qt application with overlay
    app = QtWidgets.QApplication(sys.argv)
    overlay_widget = TransparentHeatmapOverlay()
    overlay_widget.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

