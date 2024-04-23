import time

import cv2
import imutils
import numpy as np

from argparse import ArgumentParser

from tqdm import tqdm

def make_atlas(vs, max_frames=500, top=150, bot=150):
    """Makes an atlas from a video stream `vs`"""
    prev_pts = None
    prev_gray = None
    all_pts = []
    y_offsets = []
    canvas = None
    while True:
        success, frame = vs.read()
        if frame is None or not success:
            break
        frame = imutils.resize(frame, width=600)
        # crop the frame
        frame = frame[top:-bot, :]
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros((h*3, w, 3), dtype=np.uint8)
            # copy the current frame in the center of this canvas
            canvas[h*2:h*3, :] = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            all_pts.append(prev_pts.reshape(-1, 2))
            continue
        pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # convert hsv to bgr and write to a png
        cv2.imwrite(f'/tmp/atlas/flow-{len(all_pts):05d}.png', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        prev_gray = gray
        prev_pts = pts
        all_pts.append(pts.reshape(-1, 2))
        diff = np.median(all_pts[-1] - all_pts[-2], axis=0)
        #y_offsets.append(diff[1])
        # get y offset based on average optical flow in y-direction
        y_offsets.append(np.median(flow[..., 1]))
        # copy the current frame to the canvas, offset by the given y offset
        incr = -int(sum(y_offsets))
        canvas[h*2+incr:h*3+incr, :] = frame
        # save the canvas to a png with 0-padded frame number
        cv2.imwrite(f'/tmp/atlas/atlas-{len(all_pts):05d}.png', canvas)
        # also write the frame with overlaid points
        for x, y in all_pts[-1]:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            # draw a line with diffs from previous point locations
            cv2.line(frame, (int(x), int(y)), (int(x+diff[0]), int(y+diff[1])), (0, 0, 255), 1)
        cv2.imwrite(f'/tmp/atlas/points-{len(all_pts):05d}.png', frame)
        if len(all_pts) % 10 == 0:
            print(f'\rOn frame {len(all_pts)}: {diff}', end='')
        if max_frames > 0 and len(all_pts) >= max_frames:
            #print(f'{len(all_pts)}: corresponding pts: {np.hstack([all_pts[-2], all_pts[-1]])}')
            break

    return y_offsets



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_path', help='path to input video file')
    parser.add_argument('output_path', help='path to output video file')
    args = parser.parse_args()
    vs = cv2.VideoCapture(args.input_path)
    t0 = time.time()
    y_offsets = make_atlas(vs)
    print(f'Atlas took {time.time() - t0} seconds: {y_offsets}')


"""
Atlas took 24.419243097305298 seconds: [-0.0029144287, -6.439209e-05, -4.4555665e-05, 4.2114258e-05,
-0.0017788697, -4.91333e-05, -3.3569336e-06, 0.00017669678, -0.0008514404, 0.00013885499,
-0.00017456054, 0.00017425537, -0.00047485353, 0.000112609865, -9.58252e-05, 9.796143e-05,
-0.00013336181, 7.5683594e-05, -2.4414062e-06, 3.0822754e-05, 9.460449e-06, 8.239746e-06,
2.1057129e-05, 0.00015197754, 7.0495604e-05, -4.0893556e-05, -3.3569336e-06, 3.540039e-05,
-1.6174316e-05, -1.4038086e-05, -2.5024414e-05, 6.958008e-05, 6.1035156e-05, 4.91333e-05,
-2.8991699e-05, -6.713867e-06, 8.544922e-05, -2.5939942e-05, 9.368896e-05, -0.00011291504,
8.117676e-05, 3.3569337e-05, -2.746582e-05, -3.6621095e-05, -7.1716306e-05, 5.8898924e-05,
2.2457113, 6.6265955, 7.778789, 8.897106, 8.894391, 8.204284, 9.9953375, 9.528614, 8.3876505,
9.01052, 7.0182, 5.6853514, 4.4408426, 3.9982662, 4.830676, 5.1434383, 6.251884, 7.3628716,
6.3226585, 5.3053412, 7.6862373, 7.314765, 4.475904, 6.7247157, 7.666007, 4.746029, 1.8091245,
5.080952, 6.10561, 3.7232978, 1.6911944, 6.206447, 4.010136, 2.788622, -1.7633618, 4.8314996,
5.9518948, 3.7799194, 3.5888069, 1.937265, 1.8236066, -0.49581176, 1.6201495, 1.7359815, 1.4978327,
1.666034, 1.5559266, 0.48490113, 0.7837897, 0.48314452, 0.7055084, 0.4168915, 0.5153076, 0.53386045,
0.2946643, 0.23678039, 0.18638428, 0.15045838, 0.24087952, 0.0011163331, 0.2354248, -0.0006719971,
0.0044403076, -0.000244751, -5.7983398e-05, 2.7094147, 3.2151482, 6.672678, 4.4629045, 2.8528864,
4.0691504, 2.9681733, 2.495366, 4.072931, 5.5607357, 2.6175818, 2.025771, 2.1274226, 2.4163294,
2.9034631, 3.066875, 2.349577, 1.9347919, 1.7738581, 2.2026544, 4.076845, 3.1750093, 3.110705,
3.7602625, 2.4521112, 1.5455182, 2.5411549, 2.0414765, 3.212896, 2.084468, 3.788258, 3.6284297,
4.170316, 2.9110487, 3.3752747, 3.0570397, 1.3191174, 5.6025996, 2.8296704, -0.17168945, 3.2424457,
1.8632824, 0.5464777, -10.906391, 2.5349114, 4.193247, 2.3031511, 1.27203, -0.19893311, -0.94136107,
0.44702026, 0.4975879, 0.5904834, 0.32374024, 0.24446899, 0.2250598, -0.42934692, 0.18546508,
0.22889526, 0.1754541, 0.18386963, 0.14721802, 0.14686768, 0.11438233, -0.10921142, 0.12894166,
0.13111816, 0.06496582, 0.073950194, 0.0817981, 0.035098877, 0.03147583, -0.00024536133,
0.012940674, 0.00011230469, 1.5147119, 2.4643652, -0.065646976, -5.256349, -3.9429955, -4.0625477,
3.8386974, -3.4006824, 4.043014, 2.564795, 2.6629028, 3.4828076, -2.736228, 1.9790906]
"""
