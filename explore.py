from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
for i, r in enumerate(model("squat.mov", stream=True)):
    if i % 15 == 0:
        print(i, r.keypoints.xy.shape)
        print(r.keypoints.conf)