import cv2
import torch
from ultralytics import YOLO
import speech_recognition as sr
from gtts import gTTS
import tempfile
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI

# Check if MPS is available FOR MAC ONLY
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Check if CUDA is available first , fallback to CPU
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(f"Using device: {device}")

# ONLY USE CPU
#device = torch.device("cpu")

# Load the YOLOv10 Model and move it to the appropriate device
model = YOLO("yolov10m.pt")  # YOLOv10 model
model.to(device)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Set up the connection to the local LM Studio server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to speak text using gTTS and increase playback speed
def speak_text(text, speed=1.25):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        tts.save(temp_file.name)
        sound = AudioSegment.from_file(temp_file.name)
        fast_sound = sound.speedup(playback_speed=speed)
        play(fast_sound)

# Function to capture video from webcam and detect objects
def detect_objects_from_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return []

    detected_objects = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform object detection
        results = model(frame, stream=True)  # Use streaming for efficient processing

        current_detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                current_detected_objects.append(label)

        # Detect and announce new objects
        new_objects = list(set(current_detected_objects) - set(detected_objects))
        detected_objects.extend(new_objects)
        for obj in new_objects:
            speak_text(f"Detected {obj}")

        print(f"Detected objects: {detected_objects}")
        cv2.imshow("YOLOv10 - Object Detection", frame)  # Update display name

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_objects

# Function for speech recognition
def recognize_speech():
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"Recognized speech: {query}")
        return query
    except sr.UnknownValueError:
        print("Could not understand audio")
        speak_text("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        speak_text("Could not request results")
        return None

# Function to query the local LLM via LM Studio
def ask_local_llm(question, context):
    messages = [
        {"role": "system", "content": "You are an assistant that answers questions about detected objects."},
        {"role": "user", "content": f"{question}\nContext: {context}"}
    ]
    completion = client.chat.completions.create(
        model="xtuner/llava-llama-3-8b-v1_1-gguf",
        messages=messages,
        temperature=0.7,
    )
    # Return the message content directly
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Step 1: Detect objects in real-time using webcam
    detected_objects = detect_objects_from_camera()

    while True:
        # Step 2: Recognize speech
        speech_query = recognize_speech()

        if speech_query:
            # Step 3: Check if the query relates to any detected object
            relevant_object = None
            for obj in detected_objects:
                if obj.lower() in speech_query.lower():
                    relevant_object = obj
                    print(f"Question relates to detected object: {obj}")
                    speak_text(f"Question relates to detected object: {obj}")

                    # Step 4: Query the local LLM for the detected object
                    speak_text("Processing your question now.")
                    context = f"The detected object is {obj}."
                    answer = ask_local_llm(speech_query, context)

                    # Print the answer to the terminal
                    print(f"Answer: {answer}")

                    # Speak the answer out loud at a faster speed
                    speak_text(f"Answer: {answer}", speed=1.25)
                    break

            if relevant_object:
                # Keep asking for more questions until user says "that's all for today"
                while True:
                    speak_text("Do you have any more questions about the detected objects?")
                    follow_up_query = recognize_speech()

                    if follow_up_query:
                        # Check if the user wants to end the session
                        if "that's all for today" in follow_up_query.lower() or "no more questions" in follow_up_query.lower():
                            speak_text("Thank you. Have a great day!")
                            exit()  # Exit the program properly

                        # If the follow-up query is related to the detected object
                        if relevant_object.lower() in follow_up_query.lower():
                            speak_text("Processing your new question about the detected object.")
                            answer = ask_local_llm(follow_up_query, f"The detected object is {relevant_object}.")
                            print(f"Answer: {answer}")
                            speak_text(f"Answer: {answer}", speed=1.25)
                        else:
                            speak_text("The question does not relate to the detected objects.")
                    else:
                        speak_text("Sorry, I didn't catch that. Could you please repeat your question?")
            else:
                speak_text("The question does not relate to the detected objects.")
        else:
            speak_text("Sorry, I didn't catch that. Could you please repeat your question?")
