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
        
        T = cv2.cvtColor(frame0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()

        alpha = 1

        P = np.array([0,0], dtype=np.float64)
        while True:
            # Receive frame
            ret, frame = cap.read()

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not ret:
                raise Exception("Failed to Capture from device")
            

            uv = np.array([np.inf, np.inf])
            iter_counter = 0

            while np.linalg.norm(uv) > 0.5 and iter_counter < MAX_ITER:

                # Image differences: compute dIm = I(t+1, x+p) - T
                It = self.differenceThresh(T, frame_gray[int(y + P[0]) : int(y + P[0]) + h, int(x + P[1]) : int(x + P[1]) + w], 40)
                
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

            key = cv2.waitKey(1)
            if key == 27:
                break

        cv2.destroyAllWindows()
        cap.release()

        # Get rect
        # Optical flow for the bounding box: optical flow tells you direction
        # 

class HighDimensionRegTracking(TranslationalXY):
    def __init__(self, capture_source):
        super().__init__(capture_source)

    def warpIndices(self, x, y, P):
        # X, Y
        points = np.matmul(P, np.array([[x], [y], [1]])).flatten()
        return (int(points[0]), int(points[1]))
        # return (int(x + P[0, 2]), int(y + P[1, 2]))
    
    def getWarpFunction(self, P):
        return np.array([[1+P[0], P[2], P[4]], [P[1], 1+P[3], P[5]]], dtype=np.float64)
    
    def getJacobian_W(self, x, y):
        J_x = np.kron(np.arange(0, x), np.ones((y, 1)))
        J_y = np.kron(np.arange(0, y).reshape(y, 1), np.ones((1, x)))
        J_0 = np.zeros((y, x))
        J_1 = np.ones((y, x))
        return np.block([[J_x, J_0, J_y, J_0, J_1, J_0], [J_0, J_x, J_0, J_y, J_0, J_1]])
    
    def steepestDescent(self, jacobian, Ix_W, Iy_W, w, h):
        arr = []
        for i in range(6):
            Tx = np.multiply(Ix_W, jacobian[0 : h , (i*w) : (i*w)+w])
            Ty = np.multiply(Iy_W, jacobian[h:, (i*w) : (i*w)+w])
            arr.append(Tx+Ty)
        return np.array(arr)
    
    def steepestDescentUpdate(self, sd_images, diff, w):
        sd_p = np.zeros(6)
        for i in range(6):
            sd_p[i] = np.sum( np.sum( np.multiply(sd_images[i], diff) ) )
        
        return sd_p
    
    def hessian(self, sd_images, w):
        H = np.zeros((6, 6))
        for i in range(len(sd_images)):
            for j in range(len(sd_images)):
                H[i, j] = np.sum( np.sum( np.multiply(sd_images[i], sd_images[j]) ) )
        return H

    def run(self):
        cap = cv2.VideoCapture(self.capture_source)

        ret, frame0 = cap.read()

        if not ret:
            raise Exception("Failed to Capture from device")
        
        x, y, w, h = self.getrect(frame0)
        print(x, y, w, h)
        
        # Update to follow martins notes
        # https://www.youtube.com/watch?v=tzO245uWQxA
        # Affine warp (6 DOF)
        # W(x;p) = [a1 a2 b1; a3 a4 b2][x; y; 1]
        # W(x;p) = A(2x3)[x; y; 1]
        # W(x;p) = (a1x + a2y + b1, a3x + a4y + b2)
        # Jacobian of Affine transformation
        # = [1 0 x y 0 0; 0 1 0 0 x y]
        # Assume initial estimate of P is known find delta_p
        # GENERAL
        # sum(I(W(x;p+delta_p)) - T(x))^2
        # From I(W(x;p+delta_p)) find taylor series:
        # sum( I(W(x;p+delta_p)) + gradient_I * Jacobian(W) * delta_p - T(x))^2
        # Where gradient_I = [Ix, Iy]^T
        # FIND DELTA_P S.T. FUNCTION IS MINIMIZED
        # Differentiate sum
        # 2 * sum( [gradient_I * Jacobian(W)]^T * [I(W(x;p)) + gradient_I * Jacobian(W) * delta_p - T(x)]) = 0
        # delta_p = H^-1 * 2 * sum( [gradient_I * Jacobian(W)]^T * [T(x) - I(W(x;p))])
        # H^-1 = sum([gradient_I*Jacobian(W)]^T * [gradient_I*Jacobian(W)])

        cv2.destroyAllWindows()
        T = cv2.cvtColor(frame0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        P = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        # USE MESHGRID
        Xq = np.arange(x, x+w, dtype=np.float64)
        Yq = np.arange(y, y+h, dtype=np.float64)

        T_Xi, T_Yi = np.meshgrid(Xq, Yq)
        # T_XYi = np.array([T_Xi, T_Yi])
        T_XYi = np.dstack((T_Yi, T_Xi))
        print(T_XYi.shape)
        # print(T_XYi[0, 0, 0])
        # WarpedT = cv2.warpAffine(T_XYi, self.getWarpFunction(P), (T.shape[1], T.shape[0]))
                
        # p1, p2, p3, p4, p5, p6
        Jacobian_W = self.getJacobian_W(w, h)

        alpha = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            
            iter_counter = 0
            # while norm_P > 0.5 and iter_counter < MAX_ITER:
            #     idx_warped_x, idx_warped_y = self.warpIndices(x, y, P)
            #     idx_warped_x_w, idx_warped_y_h = self.warpIndices(x, y, P)
            #     It = self.differenceThresh(T, frame_gray[idx_warped_y : idx_warped_y_h, idx_warped_x : idx_warped_x_w], 40)

            #     Iy, Ix = np.gradient(np.float32(frame_gray[idx_warped_y : idx_warped_y_h, idx_warped_x : idx_warped_x_w]))
            #     # Jacobian_W = self.getJacobian_W
            #     # jacobian on corners/meshgrid points


            #     iter_counter += 1
            # Transform = [a00x + a01y + b00; a10x + a11y + b10]

            # sample = frame_gray[y : y+h, x : x+w]
            # warp = cv2.warpAffine(sample, np.array([[P[0], P[2], P[4]], [P[1], P[3], P[5]]], dtype=np.float64), (sample.shape[1], sample.shape[0]))
            print("\niter")
            norm_P = np.inf

            Iy, Ix = np.gradient(np.float32(frame_gray))

            # NEED TO GET UPDATED WARP
            while norm_P > 1 and iter_counter < MAX_ITER:
                W = self.getWarpFunction(P)

                # Warp template image POINTS
                W_T = cv2.warpAffine(T_XYi, self.getWarpFunction(P), (T.shape[1], T.shape[0]))
                print(W_T[:,0,0].astype(np.int64), W_T[0,:,1].astype(np.int64))
  
                # Select image intensities using points
                W_I = frame_gray[W_T[:,0,0].astype(np.int64), W_T[0,:,1].astype(np.int64)]

                break


                idx_warped_x, idx_warped_y = self.warpIndices(x, y, self.getWarpFunction(P))
                idx_warped_x_w, idx_warped_y_h = self.warpIndices(x+w, y+h, self.getWarpFunction(P))
                
                sample = frame_gray[int(y+P[5]):int(y+h+P[5]), int(x+P[2]):int(x+w+P[2])]
                
                warp = cv2.warpAffine(sample, self.getWarpFunction(P), (T.shape[1], T.shape[0]))
                
                diff = self.differenceThresh(warp, T, 40)

                Ix_warped = cv2.warpAffine(Ix[y: y+h, x: x+w], self.getWarpFunction(P), (T.shape[1], T.shape[0]))
                Iy_warped = cv2.warpAffine(Iy[y: y+h, x: x+w], self.getWarpFunction(P), (T.shape[1], T.shape[0]))

                sd_images = self.steepestDescent(Jacobian_W, Ix_warped, Iy_warped, w, h)
                
                inv_H = np.linalg.inv(self.hessian(sd_images, w))
                
                sd_p = self.steepestDescentUpdate(sd_images, diff, w)

                d_p = np.matmul(inv_H, sd_p)
                print(d_p)
                
                norm_P = np.linalg.norm(d_p)

                P += alpha * d_p
                print("norm: ", norm_P)

                print(self.getWarpFunction(P), '\n')
                iter_counter += 1

                
            # points = np.array([[idx_warped_x, idx_warped_y], [idx_warped_x_w, idx_warped_y], [idx_warped_x, idx_warped_y_h], [idx_warped_x_w, idx_warped_y_h]], dtype=np.int32)
            # # cv2.polylines(frame, points, True, (0, 255, 0), 1)
            # for i in points:
                # cv2.circle(frame, i, 1, (0, 255, 0), 2)
            # cv2.imshow("sample", sample)
            cv2.imshow("warp", W_I)
            # cv2.imshow("diff", diff)
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(100)
            if key == 27:
                break
        
        print("END")
        cap.release()
        cv2.destroyAllWindows()

class GaussianPyrTracking(TranslationalXY):
    def __init__(self, capture_source, levels = 3, scale = 1):
        super().__init__(capture_source)
        self.levels = levels
        self.scale = scale

    def process(self, frame_p, y_p, x_p, T_p_h, T_p_w, T_p, alpha, level_P):
        print("ENTER")
        iter_counter = 0
        d_uv = np.array([np.inf, np.inf])
        while np.linalg.norm(d_uv) > 0.3 and iter_counter < MAX_ITER:
            f_area = frame_p[int(y_p + level_P[0]) : int(y_p + level_P[0]) + T_p_h, int(x_p + level_P[1]) : int(x_p + level_P[1]) + T_p_w]
            It = self.differenceThresh(T_p, f_area, 40)
            Iy, Ix = np.gradient(np.float32(f_area))
            dI = np.array([Iy.flatten(), Ix.flatten()]).T
            It = It.flatten().T
            d_uv = np.array(np.linalg.lstsq(dI, It, rcond=None)[0])

            level_P -= alpha * d_uv

            print("norm", np.linalg.norm(d_uv))
            print("PROCESSED LEVEL_P", level_P)

            iter_counter += 1

        return level_P
    
    def run(self):
        cap = cv2.VideoCapture(self.capture_source)

        ret, frame0 = cap.read()

        if not ret:
            raise Exception("Failed to Capture from device")
        
        X = np.arange(0, frame0.shape[1], dtype=np.float32)
        Y = np.arange(0, frame0.shape[0], dtype=np.float32)
        Xq, Yq = np.meshgrid(X, Y)
        
        x, y, w, h = self.getrect(frame0)

        cv2.destroyAllWindows()

        T = cv2.cvtColor(frame0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        Tx = np.arange(0, T.shape[1], dtype=np.float32)
        Ty = np.arange(0, T.shape[0], dtype=np.float32)
        TXq, TYq = np.meshgrid(Tx, Ty)
        T = cv2.remap(T, TXq, TYq, cv2.INTER_LINEAR)

        T_pyramid = [T]
        for i in range(self.levels):
            top = T_pyramid[-1]
            for l in range(self.scale): 
                frame_ds = cv2.GaussianBlur((cv2.pyrDown(top)), (3,3), 0)
                top = frame_ds
            T_pyramid.append(top)
        T_pyramid.reverse()
        
        alpha = 0.1

        # Global disp
        P = np.array([0, 0], dtype=np.float64)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            print("\n------------------------------FRAME------------------------------\n")
            print("STARTING P: ", P)

            frame_gray_r = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame_gray_r = cv2.remap(frame_gray, Xq, Yq, cv2.INTER_LINEAR)

            uv = np.array([0, 0], dtype=np.float64)

            # INIT PYRAMIDS
            frame_pyramid = [frame_gray_r]
            for i in range(self.levels):
                top = frame_pyramid[-1]
                for l in range(self.scale): 
                    frame_ds = cv2.GaussianBlur((cv2.pyrDown(top)), (3,3), 0)
                    top = frame_ds
                
                frame_pyramid.append(top)
            frame_pyramid.reverse()
            


            for i in range(self.levels):
                print("\nLEVEL\n", i)
                T_p = T_pyramid[i]
                T_p_h, T_p_w = T_p.shape
                
                frame_p = frame_pyramid[i]

                # Propagating points
                level_P = uv + ( P // ((2 ** self.scale) ** (self.levels - i))) 
                print("STARTING LEVEL_P", level_P)

                y_i = y // (2 ** self.scale)**(self.levels - i)
                y_p = int(y_i + level_P[1])

                x_i = x // (2 ** self.scale)**(self.levels - i)
                x_p = int(x_i + level_P[0])

                lp = self.process(frame_p, y_p, x_p, T_p_h, T_p_w, T_p, alpha, level_P)

                print("FINAL LEVEL_P", level_P)
                uv = (uv + lp) * 2**self.scale

            P += lp
            print("FINAL P: ", P)
            
            cv2.rectangle(frame, (int(x + P[1]), int(y + P[0])), (int(x + P[1]) + w, int(y + P[0]) + h), (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(100)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    Txy = TranslationalXY('in.mp4')
    Txy.run()
    # Hdt = GaussianPyrTracking(0)
    # Hdt.run()