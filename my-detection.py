import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("SSD-Mobilenet-v2", threshold=0.5)

input = "/home/nvidia/jetson-inference/python/training/classification/data/train/cat/cat.14.jpg"

output = "/home/nvidia/Downloads/cat.14.jpg"

image = jetson.utils.loadImage(input)

detections = net.Detect(image)

print("Detection Results:")

for detection in detections:
     print(f"ClassID:{detection.ClassID}")
     print("Confidence:{:.6f}".format(detection.Confidence))
     print("Left:{:.6f}".format(detection.Left))
     print("Top:{:6f}".format(detection.Top))
     print("Right:{:.6f}".format(detection.Right))
     print("Bottom:{:.6f}".format(detection.Bottom))
     print("Width:{:.6f}".format(detection.Width))
     print("Height:{:.6f}".format(detection.Height))
     print("Area:{:.6f}".format(detection.Area))
     print("Center:({:.6f},{:.6f})".format(detection.Center[0],detection.Center[1]))
     print("\n")


jetson.utils.saveImage(output,image)
   
print(f"Detection image saved to {output}")

