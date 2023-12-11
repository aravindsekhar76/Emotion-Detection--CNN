from django.shortcuts import render
import cv2
import numpy as np
from django.http import StreamingHttpResponse
from .models import *
from fer import FER
import time
import webbrowser

# Create your views here.
def home(request):
    return render(request, 'home.html', {})

def contact(request):
    return render(request, 'contact.html', {})

def services(request):
    return render(request, 'services.html', {})

def blog(request):
    return render(request, 'blog.html', {})

def about(request):
    return render(request, 'about.html', {})

def getstart(request):
    return render(request, 'getstart.html', {})

def doctors(request):
    return render(request, 'doctors.html', {})

def camera(request):
    return render(request, 'camera.html', {})

def question(request):
    return render(request, 'question.html', {})

def happy(request):
    return render(request, 'happy.html', {})

def neutral(request):
    return render(request, 'neutral.html', {})

def sad(request):
    return render(request, 'sad.html', {})

def rate(request):
    return render(request, 'rate.html', {})

def angry(request):
    return render(request, 'angry.html', {})

def fear(request):
    return render(request, 'fear.html', {})

# Global variable to store the detected emotion
persistent_emotion = None

def video_stream():
    print('1')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    emotion_detector = FER(mtcnn=True)
    start_time = time.time()
    # Dictionary to store the count of each emotion
    emotion_counts = {}

    while time.time() - start_time < 5:
        ret, im = cam.read()
        analysis = emotion_detector.detect_emotions(im)

        # Process emotions for only one face (the first face detected)
        if analysis:
            objects = analysis[0]  # Get the first face detected
            bound = objects.get('box')
            emotions = objects.get('emotions')
            x, y, w, h = bound
            emotion = max(emotions, key=emotions.get)
            print('Detected Emotion:', emotion)
            # Increment the count of the detected emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
            cv2.putText(im, str(emotion), (x, y - 40), font, 1, (255, 255, 255), 3)

        # Encode the image as JPEG
        _, jpeg = cv2.imencode('.jpg', im)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cam.release()
    # Find the emotion that occurred most frequently
    global persistent_emotion
    if emotion_counts:
        persistent_emotion = max(emotion_counts, key=emotion_counts.get)
        print('Detected Emotion (Persistent):', persistent_emotion)
    # Call redirect_based_on_emotion after the while loop
    return redirect_based_on_emotion(persistent_emotion)

def redirect_based_on_emotion(persistent_emotion):
    print ("Emotion is:", persistent_emotion)
    # Check the value of the provided emotion and redirect accordingly
    if persistent_emotion == 'neutral':
        webbrowser.open('http://127.0.0.1:8000/neutral.html')
    elif persistent_emotion == 'sad':
        webbrowser.open('http://127.0.0.1:8000/sad.html')
    elif persistent_emotion == 'happy':
        webbrowser.open('http://127.0.0.1:8000/happy.html')
    elif persistent_emotion == 'angry':
        webbrowser.open('http://127.0.0.1:8000/angry.html')
    elif persistent_emotion == 'fear':
        webbrowser.open('http://127.0.0.1:8000/fear.html')  
    else:
        webbrowser.open('http://127.0.0.1:8000/getstart.html')# If no specific emotion detected or provided, redirect to a default website
    
def camera(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')
