import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Disable PyAutoGUI fail-safe mechanism (use with caution)
pyautogui.FAILSAFE = False

class VirtualMouse:
    def __init__(self):
        # Initialize MediaPipe Hand Landmarker with new Task API
        model_path = "hand_landmarker.task"
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Mouse control variables
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 7
        
        # Gesture detection thresholds
        self.click_threshold = 40
        
        # State management
        self.is_dragging = False
        self.last_click_time = 0
        self.click_cooldown = 0.3
        self.prev_scroll_y = None
        self.is_scrolling = False
        # Frame dimensions
        self.frame_w = 640
        self.frame_h = 480
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two normalized points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def get_finger_states(self, landmarks):
        """Determine which fingers are up or down"""
        if not landmarks:
            return []
        
        # Finger tip and pip landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [3, 6, 10, 14, 18]  # thumb, index, middle, ring, pinky
        
        states = []
        
        # Thumb (compare x coordinates - depends on hand orientation)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            states.append(1)  # thumb up
        else:
            states.append(0)  # thumb down
        
        # Other fingers (compare y coordinates - lower y is up in image coordinates)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                states.append(1)  # finger up
            else:
                states.append(0)  # finger down
        
        return states
    
    def detect_gesture(self, landmarks):
        """Enhanced gesture detection based on finger positions and distances"""
        if not landmarks:
            return "none"
        
        finger_states = self.get_finger_states(landmarks)
        
        # Get key landmark positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Calculate distances (normalized coordinates 0-1)
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        index_middle_dist = self.calculate_distance(index_tip, middle_tip)
        
        # Convert to pixel distance for threshold comparison
        thumb_index_px = thumb_index_dist * self.frame_w
        index_middle_px = index_middle_dist * self.frame_w
        
        # Gesture recognition logic
        # 1. LEFT CLICK (highest priority)
        if thumb_index_px < self.click_threshold:
          return "left_click"

       # 2. RIGHT CLICK / SCROLL
        elif finger_states[1] == 1 and finger_states[2] == 1:
          if index_middle_px < self.click_threshold:
           return "right_click"
          else:
           return "scroll"
     # 3. DRAG
        elif thumb_index_px < 2 * self.click_threshold:
          return "drag"

       # 4. STOP
        elif finger_states == [1, 1, 1, 1, 1]:
          return "stop"

        # 5. FIST
        elif finger_states == [0, 0, 0, 0, 0]:
         return "fist"

        # 6. CURSOR MOVEMENT (lowest priority)
        elif finger_states[1] == 1:
         return "point"

        return "none"
    
    def move_cursor(self, landmarks):
        """Smooth cursor movement based on index finger"""
        if not landmarks:
            return
        
        # Use index finger tip for cursor control (landmark 8)
        index_tip = landmarks[8]
        
        # Convert normalized coordinates (0-1) to screen coordinates
        screen_x = int(index_tip.x * self.screen_w)
        screen_y = int(index_tip.y * self.screen_h)
        
        # Apply smoothing
        curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening
        
        # Move cursor
        pyautogui.moveTo(int(curr_x), int(curr_y))
        self.prev_x, self.prev_y = curr_x, curr_y
    
    def perform_click(self, click_type):
        """Perform mouse click with cooldown"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            if click_type == "left":
                pyautogui.click()
            elif click_type == "right":
                pyautogui.rightClick()
            self.last_click_time = current_time
    
    def perform_scroll(self, landmarks):
        """Perform scrolling based on hand movement"""
        if not landmarks:
            self.prev_scroll_y = None
            self.is_scrolling = False
            return
        
        # Use middle finger tip for scrolling (landmark 12)
        middle_tip = landmarks[12]
        middle_y = int(middle_tip.y * self.frame_h)
        
        # Initialize scroll tracking when gesture starts
        if self.prev_scroll_y is None:
            self.prev_scroll_y = middle_y
            self.is_scrolling = True
            return
        
        # Calculate scroll direction and amount
        scroll_diff = middle_y - self.prev_scroll_y
        
        if abs(scroll_diff) > 5:  # Lower threshold for better responsiveness
            scroll_amount = int(scroll_diff * 2)  # More granular scrolling
            pyautogui.scroll(-scroll_amount)  # Negative for natural scrolling
        
        # Always update position for continuous tracking
        self.prev_scroll_y = middle_y
        self.is_scrolling = True
    
    def handle_drag(self, gesture, landmarks):
        """Handle drag and drop functionality"""
        if gesture == "drag" and not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True
        elif gesture != "drag" and self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
        
        # Move cursor while dragging
        if self.is_dragging and landmarks:
            self.move_cursor(landmarks)
    
    def draw_landmarks(self, img, landmarks):
        """Draw hand landmarks on image"""
        h, w, c = img.shape
        
        # Define connections between landmarks (hand skeleton)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                x1, y1 = int(start_point.x * w), int(start_point.y * h)
                x2, y2 = int(end_point.x * w), int(end_point.y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw landmark points
        for i, landmark in enumerate(landmarks):
            x, y = int(landmark.x * w), int(landmark.y * h)
            # Different colors for different fingers
            if i == 8:  # Index tip
                color = (0, 255, 0)  # Green
                radius = 10
            elif i == 4:  # Thumb tip
                color = (255, 0, 0)  # Blue
                radius = 10
            elif i == 12:  # Middle tip
                color = (0, 0, 255)  # Red
                radius = 8
            else:
                color = (255, 255, 255)  # White
                radius = 5
            cv2.circle(img, (x, y), radius, color, -1)
        
        return img
    
    def draw_interface(self, img, gesture, landmarks):
        """Draw enhanced user interface"""
        # Create semi-transparent overlay for info panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (300, 200), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Display current gesture
        gesture_text = f"Gesture: {gesture.replace('_', ' ').title()}"
        cv2.putText(img, gesture_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display status
        status = "Dragging" if self.is_dragging else "Ready"
        cv2.putText(img, f"Status: {status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        instructions = [
            "Point: Move cursor",
            "Thumb + Index: Left click",
            "Index + Middle: Right click/Scroll",
            "Thumb + Index: Drag",
            "Open Palm: Stop cursor",
            "Fist: Pause tracking",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (10, 90 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw finger states if hand is detected
        if landmarks:
            finger_states = self.get_finger_states(landmarks)
            state_text = f"Fingers: {''.join(map(str, finger_states))}"
            cv2.putText(img, state_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return img
    
    def run(self):
        """Main execution loop"""
        print("Virtual Mouse System Started!")
        print("Make sure your webcam is working")
        print("Show your hand to camera to start")
        print("Press 'q' to quit")
        
        # Test webcam connection
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam initialized successfully")
        print(f"Screen resolution: {self.screen_w}x{self.screen_h}")
        
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Failed to read from webcam")
                    time.sleep(1)  # Wait before retrying
                    continue
                
                img = cv2.flip(img, 1)  # Mirror effect
                h, w, c = img.shape
                self.frame_w, self.frame_h = w, h
                
                # Convert BGR to RGB for MediaPipe
                try:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Create MediaPipe Image object
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    
                    # Detect hand landmarks
                    detection_result = self.detector.detect(mp_image)
                    landmarks = None
                    gesture = "none"
                    
                    if detection_result.hand_landmarks:
                        # Get first hand landmarks
                        landmarks = detection_result.hand_landmarks[0]
                        
                        # Draw landmarks on image
                        img = self.draw_landmarks(img, landmarks)
                        
                        # Detect gesture
                        gesture = self.detect_gesture(landmarks)
                        print(gesture)
                        
                        # Perform actions based on gesture
                        if gesture == "point":
                            self.move_cursor(landmarks)
                        elif gesture == "left_click":
                            self.perform_click("left")
                        elif gesture == "right_click":
                            self.perform_click("right")
                        elif gesture == "scroll":
                            self.perform_scroll(landmarks)
                        elif gesture == "drag":
                            self.handle_drag(gesture, landmarks)
                        else:
                            self.handle_drag(gesture, None)
                    else:
                        self.handle_drag(gesture, None)
                        # Reset scroll state when not scrolling
                        if gesture != "scroll":
                            self.prev_scroll_y = None
                            self.is_scrolling = False
                    
                    # Draw interface
                    img = self.draw_interface(img, gesture, landmarks)
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Display frame
                cv2.imshow("Virtual Mouse - AI Hand Tracking", img)
                
                # Check for quit key (small delay for smooth performance)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nExiting Virtual Mouse...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            if self.is_dragging:
                pyautogui.mouseUp()
            print("Virtual Mouse stopped successfully")

def main():
    """Main function to initialize and run virtual mouse"""
    try:
        virtual_mouse = VirtualMouse()
        virtual_mouse.run()
    except Exception as e:
        print(f"Failed to start Virtual Mouse: {e}")
        print("Please check your webcam and dependencies")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
