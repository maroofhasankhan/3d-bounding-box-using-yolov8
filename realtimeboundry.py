import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import time

class YOLO3DBoundingBox:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.3):
        """
        Initialize YOLO model and camera settings
        """
        self.yolo_model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.camera = None
        
    def initialize_camera(self, camera_id=0, width=1280, height=720):
        """
        Initialize the camera with specified resolution
        """
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.camera.isOpened():
            raise ValueError("Could not open camera")
        
        return self.camera.isOpened()

    def process_frame(self, frame):
        """
        Process a single frame and draw 3D bounding boxes
        """
        try:
            # Convert BGR to RGB for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run YOLO detection
            results = self.yolo_model(pil_image)
            
            # Draw results on the frame
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Draw 2D bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{result.names[cls]}: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Calculate 3D box parameters based on 2D box size
                    width = x2 - x1
                    height = y2 - y1
                    depth = int(width * 0.2)  # Adaptive depth based on box width
                    
                    
                    # Draw 3D box lines
                    # Front face (green)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Top edges (magenta)
                    cv2.line(annotated_frame, (x1, y1), (x1 + depth, y1 - depth), (255, 0, 255), 2)
                    cv2.line(annotated_frame, (x2, y1), (x2 + depth, y1 - depth), (255, 0, 255), 2)
                    
                    # Bottom edges (red)
                    cv2.line(annotated_frame, (x1, y2), (x1 + depth, y2 - depth), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (x2, y2), (x2 + depth, y2 - depth), (0, 0, 255), 2)
                    
                    # Back face
                    cv2.rectangle(annotated_frame, 
                                (x1 + depth, y1 - depth),
                                (x2 + depth, y2 - depth),
                                (255, 0, 255), 2)
            
            # Add FPS counter
            fps = 1.0 / (time.time() - self.last_time) if hasattr(self, 'last_time') else 0
            self.last_time = time.time()
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return annotated_frame
            
        except Exception as e:
            print(f"Error in processing frame: {str(e)}")
            return frame

    def run_camera_detection(self):
        """
        Run real-time detection from camera feed
        """
        if not hasattr(self, 'camera') or self.camera is None:
            self.initialize_camera()
        
        self.last_time = time.time()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame and draw boxes
                result_frame = self.process_frame(frame)
                
                # Show result
                cv2.imshow("YOLO 3D Bounding Boxes (Press 'q' to quit)", result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Save frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detection_{timestamp}.jpg", result_frame)
                    print(f"Saved frame as detection_{timestamp}.jpg")
                
        finally:
            # Clean up
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()

    def release(self):
        """
        Release camera resources
        """
        if self.camera is not None:
            self.camera.release()

def main():
    # Initialize detector
    detector = YOLO3DBoundingBox(
        model_path='yolov8n.pt',  # Change to your model path if needed
        confidence_threshold=0.3
    )
    
    try:
        # Run camera detection
        detector.run_camera_detection()
        
    except KeyboardInterrupt:
        print("\nStopping detection...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        detector.release()

if __name__ == "__main__":
    main()