import numpy as np
import cv2

# frame = cv2.imread('armD32im1/1.png')
# cv2.circle(frame, (247, 157), 2, (0,255,0),1)
# cv2.circle(frame, (247+33, 157+36), 2, (0,255,0),1)
# x, y, w, h = cv2.selectROI("ROI", frame, fromCenter=False)
# print(x,y,w,h)

MAX_ITER = 50

class TranslationalXY:
    def __init__(self, capture_source):
        self.capture_source = capture_source
        self.kernel = self.lowpassInit(3)

    def lowpassInit(self, kernelSize, sigma=1.0, mu=0.0):
        x, y = np.meshgrid(np.linspace(-1, 1, kernelSize), np.linspace(-1, 1, kernelSize))
        d = np.sqrt(x*x + y*y)
        kernel = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
        return kernel
    
    def getrect(self, frame):
        return cv2.selectROI("ROI", frame, fromCenter=False)
    
    def lowpassFilter(self, img):
        smoothed = cv2.filter2D(img, -1, self.kernel)
        return smoothed
    
    def differenceThresh(self, img1, img2, threshold = 0):
        A = np.float32(img2) - np.float32(img1)
        A[np.abs(A) <= threshold] = 0
        return A
    
    def run(self):
        cap = cv2.VideoCapture(self.capture_source)

        ret, frame0 = cap.read()

        if not ret:
            raise Exception("Failed to Capture from device")

        x, y, w, h = self.getrect(frame0)
        print(x, y, w, h)
        
        T = cv2.cvtColor(frame0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()

        alpha = 1

        # Init p = 0

        P = np.array([0,0], dtype=np.float64)
        while True:
            # Receive frame
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("\niter\n")

            diff = 0

            if not ret:
                raise Exception("Failed to Capture from device")
            
            # TEST
            uv = np.array([np.inf, np.inf])
            # TEST

            iter_counter = 0

            while np.linalg.norm(uv) > 0.5 and iter_counter < MAX_ITER:

                # Image differences: compute dIm = I(t+1, x+p) - T
                It = self.differenceThresh(T, frame_gray[int(y + P[0]) : int(y + P[0]) + h, int(x + P[1]) : int(x + P[1]) + w], 40)
                diff = It
                
                # Image gradients
                Iy, Ix = np.gradient(np.float32(frame_gray[int(y + P[0]) : int(y + P[0]) + h, int(x + P[1]) : int(x + P[1]) + w]))
                dI = np.array([Iy.flatten(), Ix.flatten()]).T
                It = It.flatten().T

                # Least squares
                uv = np.array(np.linalg.lstsq(dI, It)[0])

                P -= alpha * uv

                iter_counter += 1
                
                        
            cv2.rectangle(frame, (int(x + P[1]), int(y + P[0])), (int(x + P[1]) + w, int(y + P[0]) + h), (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(100)
            if key == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

        # Get rect
        # Optical flow for the bounding box: optical flow tells you direction
        # 

if __name__ == "__main__":
    Txy = TranslationalXY('in.mp4')
    Txy.run()