import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
from flask import Flask, json, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 

UPLOAD_FOLDER = 'C:\\Users\\prasa\\Desktop\\Flask'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 

@app.route('/')
def main():
    return render_template('sample.html')
 
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == "POST":
        if request.files:
            vid = request.files['video']
            print(vid)
            vid.save(os.path.join(app.config['UPLOAD_FOLDER'], vid.filename))
            
            ###############################

            # CAM = 0

            # print('SELECT OPTION FROM BELOW')
            # print('1. Input video through camera')
            # print('2. sample video')

            # response = int(input())

            # # default resourcs
            # res = CAM

            # if response == 1:
            #     res = CAM

            # elif response == 2:
            #     # path = input()
            #     VID_PATH = '1.mp4'
            #     res = VID_PATH
            # else:
            #     print('Not proper input')

            # print('click \'Q\' to quit')

            # ######## video input module #########


            video = cv2.VideoCapture(vid.filename)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            threshold = 20

            # print(width, height) # for current video 640 X 360

            writer = cv2.VideoWriter(
                'final.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

            ret, first_frame = video.read()  # first frame form input video

            prev_frame = first_frame

            unique_frames = 0
            common_frames = 0
            total_frames = 0


            protopath = 'MobileNetSSD_deploy.prototxt'     # network file
            modelpath = 'MobileNetSSD_deploy.caffemodel'   # the network weights file


            # prototxt   -> path to the .prototxt file with text description of the network architecture.
            # caffeModel ->	path to the .caffemodel file with learned network.


            # Caffe is a deep learning framework made with expression, speed, and modularity in mind.

            # loading of caffe model
            detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

            CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]

            frame_number = 1

            while True:
                ret, frame = video.read()
                (H, W) = frame.shape[:2]

                # convert image into binary large object
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

                detector.setInput(blob)

                # perform classification
                person_detections = detector.forward()
                # print(person_detections.shape[2])

                # to keep track of number of frames
                # print(frame_number)
                frame_number += 1

                for i in np.arange(0, person_detections.shape[2]):

                    confidence = person_detections[0, 0, i, 2]
                    # print(confidence)

                    if confidence > 0.5:

                        # class label (id)
                        idx = int(person_detections[0, 0, i, 1])

                        # if object found is not person then continue
                        if CLASSES[idx] != "person":
                            continue

                        if (((np.sum(np.absolute(frame - prev_frame)) / np.size(frame)) > threshold)):
                            writer.write(frame)
                            prev_frame = frame
                            unique_frames += 1

                        else:
                            prev_frame = frame
                            common_frames += 1
                    # print('no person')

                cv2.imshow("Application", frame)
                total_frames += 1

                key = cv2.waitKey(1)

                if key == ord('q'):
                    break

            print("Total frames: ", total_frames)
            print("Unique frames: ", unique_frames)
            print("Common frames: ", common_frames)
            print('done')
            video.release()
            writer.release()
            cv2.destroyAllWindows()

            return 'uploaded'
    
    return 'not uploaded'
    
if __name__ == '__main__':
    app.run(debug=True)