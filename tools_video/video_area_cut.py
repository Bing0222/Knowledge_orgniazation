import cv2
import argparse
import time

parser = argparse.ArgumentParser(
    description='Cut the area of a video.'
)
parser.add_argument('--input', required=True, help='video to cut.')
parser.add_argument('--output', default='cut_video.avi', help='path to output video.')
parser.add_argument('--width', type=int, default=450, help='width of target video.')
parser.add_argument('--height', type=int, default=420, help='height of target video.')
parser.add_argument('--fps', type=float, default=20.0, help='the FPS of target video.')

args = parser.parse_args()



cap = cv2.VideoCapture(args.input)


cap = cv2.VideoCapture(args.input)
# 获取源视频的长宽
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 确定目标视频长宽
width = args.width if args.width <= w else w
height = args.height if args.height <= h else h


x1 = int((w - width) / 2)
y1 = int((h - height) / 2) - 20
x2 = x1 + width
y2 = y1 + height   # 420
size = (x2 - x1, y2 - y1)
print(size)


# vedio writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, args.fps, size)


print('Start cut...')
start_time = time.time()


i = 0
while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    cropImg = frame[y1:y2, x1:x2]  # 获取感兴趣区域
    out.write(cropImg)
    i += 1

end_time = time.time()
cap.release()
out.release()
print('total{}fps,time is{:.2f}s'.format(i, end_time - start_time))
