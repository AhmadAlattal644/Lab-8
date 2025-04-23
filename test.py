import cv2
import numpy as np
import os

SETTINGS = {
    'ref_image': 'ref-point.jpg',
    'fly_image': 'fly64.png',
    'marker_dict': cv2.aruco.DICT_4X4_50,
    'blur_kernel': (15, 15),
    'output_coords': 'coordinates.txt'
}

class ArucoTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Ошибка камеры")
        
        self.fly_img = cv2.imread(SETTINGS['fly_image'], cv2.IMREAD_UNCHANGED)
        if self.fly_img is not None and max(self.fly_img.shape[:2]) > 100:
            self.fly_img = cv2.resize(self.fly_img, (64, 64))
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(SETTINGS['marker_dict'])
        self.detector_params = cv2.aruco.DetectorParameters()
        
        open(SETTINGS['output_coords'], 'w').close()

    def process_reference(self):
        if os.path.exists(SETTINGS['ref_image']):
            img = cv2.imread(SETTINGS['ref_image'])
            blurred = cv2.GaussianBlur(img, SETTINGS['blur_kernel'], 0)
            cv2.imwrite('blurred_' + SETTINGS['ref_image'], blurred)

    def save_coords(self, x, y):
        with open(SETTINGS['output_coords'], 'a') as f:
            f.write(f"{x},{y}\n")

    def process_frame(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.detector_params)
        
        if ids is not None:
            x, y = np.mean(corners[0][0], axis=0).astype(int)
            self.save_coords(x, y)
            
            if self.fly_img is not None:
                h, w = self.fly_img.shape[:2]
                y1, y2 = y-h//2, y+h//2
                x1, x2 = x-w//2, x+w//2
                
                if 0 <= y1 < y2 <= frame.shape[0] and 0 <= x1 < x2 <= frame.shape[1]:
                    if self.fly_img.shape[2] == 4:
                        alpha = self.fly_img[:, :, 3]/255.0
                        for c in range(3):
                            frame[y1:y2, x1:x2, c] = (1-alpha)*frame[y1:y2, x1:x2, c] + alpha*self.fly_img[:, :, c]
                    else:
                        frame[y1:y2, x1:x2] = self.fly_img
            
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        
        return frame

    def run(self):
        try:
            self.process_reference()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                cv2.imshow('Трекинг маркера', self.process_frame(frame))
                
                if cv2.waitKey(1) == ord('q'):
                    break
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    ArucoTracker().run()