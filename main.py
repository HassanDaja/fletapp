import flet as ft
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import threading
import base64
from io import BytesIO
import time

# Load the TFLite models for face classification and detection
classifier_interpreter = tf.lite.Interpreter(model_path="assets/face_position_classifier.tflite")
classifier_interpreter.allocate_tensors()
input_details = classifier_interpreter.get_input_details()
output_details = classifier_interpreter.get_output_details()

yolo_interpreter = tf.lite.Interpreter(model_path='assets/face_detection_yolov5s.tflite')
yolo_interpreter.allocate_tensors()
yolo_input_details = yolo_interpreter.get_input_details()
yolo_output_details = yolo_interpreter.get_output_details()

# Define image size for classification model
img_height, img_width = 224, 224
class_names = ['front', 'left', 'right']

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Variables to store detection state
frontal_face_found = False
left_profile_found = False
right_profile_found = False
curr_phase = 0
cropped_faces = []


def classify_face_position(face_img):
    face_img = face_img.resize((img_width, img_height))
    face_array = np.array(face_img).astype(np.float32) / 255.0
    input_data = np.expand_dims(face_array, axis=0)

    classifier_interpreter.set_tensor(input_details[0]['index'], input_data)
    classifier_interpreter.invoke()
    results = classifier_interpreter.get_tensor(output_details[0]['index'])

    position_class = np.argmax(results, axis=-1)[0]
    confidence = results[0][position_class]
    return position_class, confidence


def detect_faces(page: ft.Page):
    global frontal_face_found, left_profile_found, right_profile_found, curr_phase, cropped_faces

    # Create an Flet Image widget and add it to the page once
    image_widget = ft.Image(width=400, height=700)
    page.add(image_widget)

    while cap.isOpened() and not (frontal_face_found and left_profile_found and right_profile_found):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_shape = yolo_input_details[0]['shape']
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        frame_normalized = frame_resized / 255.0
        input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)

        yolo_interpreter.set_tensor(yolo_input_details[0]['index'], input_data)
        yolo_interpreter.invoke()
        boxes = yolo_interpreter.get_tensor(yolo_output_details[0]['index'])[0]

        h, w, _ = frame.shape
        confidence_threshold = 0.7
        filtered_boxes = []
        filtered_scores = []

        for box in boxes:
            x_center, y_center, width, height, objectness_score, classcode = box
            if objectness_score > confidence_threshold:
                xmin = int((x_center - width / 2) * w)
                ymin = int((y_center - height / 2) * h)
                xmax = int((x_center + width / 2) * w)
                ymax = int((y_center + height / 2) * h)
                filtered_boxes.append([xmin, ymin, xmax, ymax])
                filtered_scores.append(float(objectness_score))

        indices = cv2.dnn.NMSBoxes(filtered_boxes, filtered_scores, confidence_threshold, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                xmin, ymin, xmax, ymax = filtered_boxes[i]
                cropped_face = frame[ymin:ymax, xmin:xmax]
                face_img = Image.fromarray(cropped_face)

                position_class, confidence = classify_face_position(face_img)

                if curr_phase == 0 and position_class == 0 and not frontal_face_found:
                    frontal_face_found = True
                    cropped_faces.append(face_img)
                    curr_phase += 1
                    page.add(ft.Text("Frontal face detected."))

                elif curr_phase == 1 and position_class == 1 and not left_profile_found:
                    left_profile_found = True
                    cropped_faces.append(face_img)
                    curr_phase += 1
                    page.add(ft.Text("Left profile detected."))

                elif curr_phase == 2 and position_class == 2 and not right_profile_found:
                    right_profile_found = True
                    cropped_faces.append(face_img)
                    page.add(ft.Text("Right profile detected."))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_names[position_class]} ({confidence:.2f})", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Convert frame to JPEG format and base64 encode it
        frame_pil = Image.fromarray(frame)
        buffer = BytesIO()
        frame_pil.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        # Encode the image to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Update the image widget's source in real-time
        image_widget.src_base64 = image_base64
        page.update()

        # Add a small delay to control the frame rate
        time.sleep(0.01)  # ~30 FPS

        # Once all faces are detected, redirect to another page
        if frontal_face_found and left_profile_found and right_profile_found:
            time.sleep(1)  # Add a small delay before redirecting
            page.add(ft.Text("All faces detected! Redirecting..."))
            page.update()
            time.sleep(1)  # Wait a moment for the user to see the text
            show_cropped_faces(page)


def show_cropped_faces(page: ft.Page):
    page.controls.clear()  # Clear the current page
    page.add(ft.Text("All faces detected! Showing the cropped faces..."))

    # Display all the cropped face images
    for idx, face_img in enumerate(cropped_faces):
        buffer = BytesIO()
        face_img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        page.add(ft.Image(src_base64=base64.b64encode(image_data).decode(), width=150, height=150))

    page.update()


def main(page: ft.Page):
    page.title = "Face Detection and Classification"
    page.add(ft.Text("Position your face in front of the camera..."))
    threading.Thread(target=detect_faces, args=(page,)).start()


ft.app(target=main,assets_dir="assets")
