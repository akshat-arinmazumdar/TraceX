from flask import Flask, Response
from flask_cors import CORS  # Add this import
import cv2

# Load the model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
file_name = 'labels.names'
with open(file_name, 'rt') as fpt:
    Class_Labels = fpt.read().rstrip('\n').split('\n')

# Configure model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use webcam instead of YouTube URL (YouTube doesn't work directly with OpenCV)
# Set camera resolution before the loop
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Set width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Or use a test video file instead:
# camera = cv2.VideoCapture('test_video.mp4')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break
        else:
            try:
                # Resize frame to match w-full h-96 (adjust aspect ratio)
                frame = cv2.resize(frame, (1440, 464))  # w-full h-96 equivalent
                
                # Object detection
                ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

                if len(ClassIndex) != 0:
                    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                        if ClassInd - 1 < len(Class_Labels):  # Safety check

                            x,y,w,h = boxes
                            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y+h)), (255, 0, 0), 2)
                            cv2.putText(frame, Class_Labels[ClassInd - 1],
                                        (boxes[0] + 10, boxes[1] + 40),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in frame processing: {e}")
                continue

@app.route('/')
def index():
    return """
    <h2>Object Detection Live Stream</h2>
    <div style='width:100%;height:384px;'>
        <img src='/video_feed' style='width:100%;height:100%;object-fit:cover;' />
    </div>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Access-Control-Allow-Origin': '*',
                       'Access-Control-Allow-Methods': 'GET',
                       'Access-Control-Allow-Headers': 'Content-Type'
                   })

if __name__ == "__main__":
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera")
        # Try alternative camera index
        camera = cv2.VideoCapture(1)
    
    app.run(debug=True, host='127.0.0.1', port=5000)