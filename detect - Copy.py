import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations
from PIL import Image
import argparse


# print("Check for the run in subprocess from detect route")

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Video source file name')
# Parse command-line arguments
args = parser.parse_args()


# load the COCO class labels our YOLO model was trained on
# labelsPath = "coco.names"
labelsPath = "DetectPY\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
# weightsPath = "yolov3.weights"
# configPath = "yolov3.cfg"
weightsPath = "DetectPY\yolov3.weights"
configPath = "DetectPY\yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# lb = joblib.load('lb.pkl')
lb = joblib.load('DetectPY\lb.pkl')
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print('Loading model and label binarizer...')
# lb = joblib.load('lb.pkl')
lb = joblib.load('DetectPY\lb.pkl')

model = CustomCNN()
print('Model Loaded...')
# model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.load_state_dict(torch.load('DetectPY\model.pth', map_location='cpu'))
model.eval()
print('Loaded model state_dict...')
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])

def detectDrowning(source):
    (W, H) = (None, None)
    print(source)
    # vs = cv2.VideoCapture('videos/'+source)
    vs = cv2.VideoCapture(source)
    while True:
        # read the next frame from the file
        grabbed, frame = vs.read()
        
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
                        (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > 0.5:
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                # actually returns the center (x, y)-coordinates of
                                # the bounding box followed by the boxes' width and
                                # height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        model.eval()
                        with torch.no_grad():
                                                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                                        pil_image = aug(image=np.array(pil_image))['image']

                                                        pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                                                        pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                                                        pil_image = pil_image.unsqueeze(0)
                                                        outputs = model(pil_image)
                                                        _, preds = torch.max(outputs.data, 1)

                                                        print("Swimming status : ",lb.classes_[preds])
                                                        if(lb.classes_[preds] =='drowning'):
                                                                        isDrowning = True
                                                        if(lb.classes_[preds] =='normal'):
                                                                        isDrowning = False
                                                                        
                                                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                                        cv2.putText(frame, lb.classes_[preds], (x, y - 5),
                                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the file pointers
    print("[INFO] cleaning up...")
    #writer.release()
    vs.release()
    cv2.destroyAllWindows()


detectDrowning(args.source)



