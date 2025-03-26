import cv2
import numpy as np
import pyautogui as gui
import time
import argparse

# Set keypress delay to 0.
gui.PAUSE = 0

# Global variables
last_mov = ''
control_mode = 'face'  # Default control mode: 'face' or 'hand'
is_paused = False
show_controls = True
bbox_size = 150  # Default box radius (half-width)
bbox_height = 200  # Default box half-height
calibration_mode = False
sensitivity = 1.0  # Control sensitivity
game_click_pos = (500, 500)  # Default position to click for game start

# Loading the pre-trained face model
model_path = './model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = './model/deploy.prototxt'

# Load hand detection model (using mediapipe if available, otherwise use a simplified approach)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mediapipe_available = True
    print("MediaPipe loaded successfully for hand detection")
except ImportError:
    mediapipe_available = False
    print("MediaPipe not available. Using simplified hand detection.")


def detect_face(net, frame):
    '''
    Detect the faces in the frame.

    returns: list of faces in the frame
                here each face is a dictionary of format-
                {'start': (startX,startY), 'end': (endX,endY), 'confidence': confidence}
    '''
    detected_faces = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({
                'start': (startX, startY),
                'end': (endX, endY),
                'confidence': confidence})
    return detected_faces


def detect_hand(frame):
    '''
    Detect hands in the frame using MediaPipe or color-based detection as fallback.
    
    returns: Hand position (x, y) or None if no hand detected
    '''
    if mediapipe_available:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the center of the hand (using the wrist position)
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)
                cy = int(hand_landmarks.landmark[0].y * h)
                
                # Draw a circle at the center of the hand
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                
                return (cx, cy)
    else:
        # Simple color-based hand detection (not as accurate but works without mediapipe)
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Only proceed if the contour is large enough
            if cv2.contourArea(max_contour) > 5000:
                # Get the centroid
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw the hand contour and center
                    cv2.drawContours(frame, [max_contour], 0, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                    
                    return (cx, cy)
    
    return None


def draw_face(frame, detected_faces):
    '''
    Draw rectangular box over detected faces.

    returns: frame with rectangular boxes over detected faces.
    '''
    for face in detected_faces:
        cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 2)
    return frame


def check_rect(detected_objects, bbox, frame_center):
    '''
    Check if detected object (face or hand) is inside the bounding box at center.

    returns: True or False.
    '''
    left_x, right_x, bottom_y, top_y = bbox
    
    if control_mode == 'face':
        for face in detected_objects:
            x1, y1 = face['start']
            x2, y2 = face['end']
            if x1 > left_x and x2 < right_x:
                if y1 > top_y and y2 < bottom_y:
                    return True
    elif control_mode == 'hand' and detected_objects:
        cx, cy = detected_objects
        if left_x < cx < right_x and top_y < cy < bottom_y:
            return True
            
    return False


def move(detected_objects, bbox, frame_center):
    '''
    Press correct button depending on position of detected object and bbox.
    Enhanced with proportional control based on distance from center.
    '''
    global last_mov
    
    if is_paused:
        return
        
    # Center position of the frame
    center_x, center_y = frame_center
    
    # Control based on face or hand
    if control_mode == 'face':
        for face in detected_objects:
            x1, y1 = face['start']
            x2, y2 = face['end']
            
            # Calculate face center
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            
            # Center
            if check_rect([face], bbox, frame_center):
                last_mov = 'center'
                return
            
            # Distance from center (for proportional control)
            x_distance = abs(face_center_x - center_x) / center_x
            y_distance = abs(face_center_y - center_y) / center_y
            
            # Scale by sensitivity
            x_distance *= sensitivity
            y_distance *= sensitivity
            
            # Apply controls based on position
            if last_mov == 'center':
                # Left
                if x1 < bbox[0]:
                    # Repeat key presses based on distance for smoother control
                    presses = max(1, int(x_distance * 3))
                    for _ in range(presses):
                        gui.press('left')
                    last_mov = 'left'
                # Right
                elif x2 > bbox[1]:
                    presses = max(1, int(x_distance * 3))
                    for _ in range(presses):
                        gui.press('right')
                    last_mov = 'right'
                # Down
                if y2 > bbox[2]:
                    presses = max(1, int(y_distance * 3))
                    for _ in range(presses):
                        gui.press('down')
                    last_mov = 'down'
                # Up
                elif y1 < bbox[3]:
                    presses = max(1, int(y_distance * 3))
                    for _ in range(presses):
                        gui.press('up')
                    last_mov = 'up'
                
                # Print out the button pressed if any
                if last_mov != 'center':
                    print(last_mov)
    
    elif control_mode == 'hand' and detected_objects:
        hand_x, hand_y = detected_objects
        
        # Check if hand is in center rectangle
        if left_x < hand_x < right_x and top_y < hand_y < bottom_y:
            last_mov = 'center'
            return
            
        # Distance from center (for proportional control)
        x_distance = abs(hand_x - center_x) / center_x
        y_distance = abs(hand_y - center_y) / center_y
        
        # Scale by sensitivity
        x_distance *= sensitivity
        y_distance *= sensitivity
        
        # Apply controls based on hand position
        if last_mov == 'center':
            # Left
            if hand_x < bbox[0]:
                presses = max(1, int(x_distance * 3))
                for _ in range(presses):
                    gui.press('left')
                last_mov = 'left'
            # Right
            elif hand_x > bbox[1]:
                presses = max(1, int(x_distance * 3))
                for _ in range(presses):
                    gui.press('right')
                last_mov = 'right'
            # Down
            if hand_y > bbox[2]:
                presses = max(1, int(y_distance * 3))
                for _ in range(presses):
                    gui.press('down')
                last_mov = 'down'
            # Up
            elif hand_y < bbox[3]:
                presses = max(1, int(y_distance * 3))
                for _ in range(presses):
                    gui.press('up')
                last_mov = 'up'
            
            # Print out the button pressed if any
            if last_mov != 'center':
                print(last_mov)


def draw_ui_elements(frame, bbox, fps):
    '''
    Draw UI elements on the frame for better user feedback
    '''
    left_x, right_x, bottom_y, top_y = bbox
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2
    
    # Draw control rectangle
    frame = cv2.rectangle(
        frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 2)
        
    # Draw directional indicators when out of center
    if last_mov == 'left':
        cv2.arrowedLine(frame, (center_x, center_y), (center_x - 50, center_y), (255, 0, 0), 3)
    elif last_mov == 'right':
        cv2.arrowedLine(frame, (center_x, center_y), (center_x + 50, center_y), (255, 0, 0), 3)
    elif last_mov == 'up':
        cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y - 50), (255, 0, 0), 3)
    elif last_mov == 'down':
        cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y + 50), (255, 0, 0), 3)
        
    # Show FPS counter
    cv2.putText(frame, f"FPS: {fps}", (frame_width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show current control mode
    cv2.putText(frame, f"Mode: {control_mode.capitalize()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show pause status if paused
    if is_paused:
        cv2.putText(frame, "PAUSED", (center_x - 70, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show calibration mode if active
    if calibration_mode:
        cv2.putText(frame, "Calibration Mode", (center_x - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press + / - to adjust size", (center_x - 120, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show controls help if enabled
    if show_controls:
        cv2.putText(frame, "Controls:", (10, frame_height - 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "P: Pause/Resume", (20, frame_height - 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "C: Calibration mode", (20, frame_height - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "M: Switch control mode", (20, frame_height - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "H: Hide this help", (20, frame_height - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "S: Adjust sensitivity", (20, frame_height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Esc: Exit", (20, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


def play(prototxt_path, model_path):
    '''
    Run the main loop until cancelled.
    Enhanced with additional features for usability.
    '''
    global last_mov, control_mode, is_paused, show_controls, bbox_size, bbox_height
    global calibration_mode, sensitivity, game_click_pos
    
    # Used to record the time when we processed last frame
    prev_frame_time = 0
    # Used to record the time at which we processed current frame
    new_frame_time = 0

    # Load the face detection model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Handle camera initialization error
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        attempts = 0
        while not cap.isOpened() and attempts < 5:
            attempts += 1
            print(f"Attempt {attempts}/5 to reconnect camera...")
            cap = cv2.VideoCapture(0)
            time.sleep(1)
        
        if not cap.isOpened():
            print("Failed to initialize camera after multiple attempts. Exiting.")
            return

    # Counter for skipping frame
    count = 0

    # Used to initialize the game
    init = 0

    # Getting the Frame width and height
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    
    # Calculate frame center
    frame_center = (frame_width // 2, frame_height // 2)

    # Initial configuration for bounding box
    left_x, top_y = frame_width // 2 - bbox_size, frame_height // 2 - bbox_height
    right_x, bottom_y = frame_width // 2 + bbox_size, frame_height // 2 + bbox_height
    bbox = [left_x, right_x, bottom_y, top_y]

    print("Game control initialized successfully!")
    print("Position your face or hand in the center rectangle to start the game.")
    print("Controls:")
    print("  P: Pause/Resume the game")
    print("  C: Enter calibration mode (adjust control box size)")
    print("  M: Switch between face and hand control modes")
    print("  H: Show/hide help text")
    print("  S: Adjust sensitivity")
    print("  Esc: Exit")

    while True:
        # Reset FPS for this frame
        fps = 0
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to retrieve frame from camera. Retrying...")
            time.sleep(0.5)
            continue

        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Update bounding box if in calibration mode
        if calibration_mode:
            left_x, top_y = frame_width // 2 - bbox_size, frame_height // 2 - bbox_height
            right_x, bottom_y = frame_width // 2 + bbox_size, frame_height // 2 + bbox_height
            bbox = [left_x, right_x, bottom_y, top_y]
        
        # Detect objects based on control mode
        if control_mode == 'face':
            # Detect faces
            detected_objects = detect_face(net, frame)
            # Draw bounding box around detected faces
            frame = draw_face(frame, detected_objects)
        else:  # Hand mode
            # Detect hand position
            detected_objects = detect_hand(frame)
        
        # Draw UI elements (control rectangle, indicators, etc.)
        frame = draw_ui_elements(frame, bbox, fps)

        # Only process controls every other frame to reduce CPU usage
        if count % 2 == 0 and not is_paused:
            # For first pass - initialize the game
            if init == 0:
                # Check if object is inside the control rectangle
                if (control_mode == 'face' and detected_objects and check_rect(detected_objects, bbox, frame_center)) or \
                   (control_mode == 'hand' and detected_objects and check_rect(detected_objects, bbox, frame_center)):
                    init = 1
                    cv2.putText(
                        frame, 'Game is running', (frame_width // 2 - 100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.waitKey(10)
                    last_mov = 'center'
                    # Click to start the game at configured position
                    gui.click(x=game_click_pos[0], y=game_click_pos[1])
            else:
                # Process movement controls
                move(detected_objects, bbox, frame_center)
                cv2.waitKey(50)

        # Calculate FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # Display the frame
        cv2.imshow('Enhanced Game Controller', frame)
        count += 1

        # Handle key presses
        k = cv2.waitKey(5)
        if k == 27:  # Esc key
            break
        elif k == ord('p') or k == ord('P'):  # Pause/Resume
            is_paused = not is_paused
            print("Game", "paused" if is_paused else "resumed")
        elif k == ord('c') or k == ord('C'):  # Calibration mode
            calibration_mode = not calibration_mode
            print("Calibration mode", "enabled" if calibration_mode else "disabled")
        elif k == ord('m') or k == ord('M'):  # Switch control mode
            control_mode = 'hand' if control_mode == 'face' else 'face'
            print(f"Switched to {control_mode} control mode")
        elif k == ord('h') or k == ord('H'):  # Toggle help text
            show_controls = not show_controls
        elif k == ord('+') or k == ord('='):  # Increase box size
            if calibration_mode:
                bbox_size = min(bbox_size + 10, frame_width // 2 - 20)
                bbox_height = min(bbox_height + 10, frame_height // 2 - 20)
        elif k == ord('-') or k == ord('_'):  # Decrease box size
            if calibration_mode:
                bbox_size = max(bbox_size - 10, 50)
                bbox_height = max(bbox_height - 10, 50)
        elif k == ord('s') or k == ord('S'):  # Adjust sensitivity
            sensitivity = (sensitivity + 0.25) % 2.0
            if sensitivity == 0:
                sensitivity = 0.25
            print(f"Sensitivity set to {sensitivity:.2f}")
        
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Game controller closed.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Enhanced Web Game Controller')
    parser.add_argument('--mode', choices=['face', 'hand'], default='face',
                        help='Control mode: face or hand')
    parser.add_argument('--click-pos', nargs=2, type=int, default=[500, 500],
                        metavar=('X', 'Y'), help='Position to click for game start (default: 500 500)')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Control sensitivity (0.25-2.0, default: 1.0)')
    parser.add_argument('--box-size', type=int, default=150,
                        help='Initial control box width (pixels from center, default: 150)')
    parser.add_argument('--box-height', type=int, default=200,
                        help='Initial control box height (pixels from center, default: 200)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set global variables from arguments
    control_mode = args.mode
    game_click_pos = tuple(args.click_pos)
    sensitivity = args.sensitivity
    bbox_size = args.box_size
    bbox_height = args.box_height
    
    # Initialize the last movement
    last_mov = ''
    
    # Start the game controller
    play(prototxt_path, model_path)
