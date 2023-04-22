import cv2
import argparse
import time

parser = argparse.ArgumentParser(
    description='This is a script to change the FPS of a video.')
parser.add_argument('--input',required=True, help='Path to video to modify.')
parser.add_argument('--fps',type=float, default=20.0,help='The target FPS you need.')
parser.add_argument('--output',default='output_20fps.avi', help='Path to output video.')

args = parser.parse_args()

cap = cv2.VideoCapture(args.input)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
size = (w,h)
out = cv2.VideoWriter(args.output,fourcc,fps=args.fps,frameSize=size)

print("Start change...")
start_time = time.time()
i = 0
while True:
    hasFrame,frame = cap.read()
    if not hasFrame:
        break
    out.write(frame)
    i += 1

end_time = time.time()
cap.release()
out.release()
print("Reading {} fps, time uses{:.2f}s".format(i,end_time-start_time))

