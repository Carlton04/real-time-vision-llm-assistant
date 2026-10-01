Real-Time Vision + Local LLM Assistant

A real-time computer vision and voice-interaction system combining YOLOv10 object detection, speech recognition, text-to-speech, and a locally hosted LLM through LM Studio.

The project was developed as part of an MSc research project exploring the integration of computer vision with language-based interaction.

Demo

▶️ Watch the project demo on YouTube - https://youtu.be/F6J9RCXHqxQ

⸻

Overview

This project combines several AI components into an interactive pipeline:

1. A camera captures the surrounding environment.
2. A fine-tuned YOLOv10m model detects objects in real time.
3. Detected object labels are stored as contextual information.
4. The user asks questions using speech.
5. Speech is converted into text.
6. The system identifies whether the question relates to a detected object.
7. The question and detected-object context are sent to a local LLM running through LM Studio.
8. The generated response is converted back into speech.

The current implementation uses the LLM primarily for object-grounded language interaction. The raw camera image is processed by YOLOv10; the LLM receives the detected object label as context rather than the camera image itself.

System Architecture

Stage	Component	Output
1	Camera	Live video frames
2	YOLOv10m	Detected objects and bounding boxes
3	Context Generation	Detected-object context
4	Speech Recognition	User’s spoken question converted to text
5	Object Relevance Check	Determines whether the question relates to a detected object
6	Local LLM via LM Studio	Generates an answer using the detected-object context
7	Text-to-Speech	Converts the generated answer into speech
8	Audio Output	Spoken response to the user

Pipeline

Camera → YOLOv10m → Detected Objects → Speech Recognition → Object Relevance Check → Local LLM → Text-to-Speech → Audio Output

Key Technical Components

Computer Vision

* YOLOv10m for real-time object detection
* Fine-tuned on the SKU-110K retail object detection dataset
* OpenCV for camera capture and visualisation
* Bounding-box visualisation during inference
* GPU/device-aware execution using Apple Silicon MPS when available

Voice Interaction

* SpeechRecognition for speech-to-text
* Google speech recognition service through the SpeechRecognition library
* gTTS for text-to-speech
* pydub for audio playback and speech-speed adjustment
* Voice-based follow-up questions

Local Language Model

* LM Studio used to host the local LLM
* OpenAI-compatible local API interface
* Model used by the current implementation:

xtuner/llava-llama-3-8b-v1_1-gguf

The LLM is provided with the user’s question together with detected-object context.

Important: despite the model name containing “LLaVA”, the current implementation does not send the raw camera image to the LLM. Visual understanding is performed by YOLOv10, and the resulting object labels are passed to the language model.

Engineering

* Modular project structure
* Git/GitHub version control
* Environment and model-weight exclusion through .gitignore
* Evaluation artifacts from model training
* Local inference architecture
* Device-aware PyTorch execution

⸻

Model Training

The object detection model was fine-tuned using the SKU-110K dataset.

Parameter	Configuration
Base model	YOLOv10m
Dataset	SKU-110K
Training epochs	20
Image size	640 × 640
Batch size	8
Device	Apple MPS when available, otherwise CPU
Workers	2

The trained model weights are intentionally not included in this repository.

⸻

Evaluation Results

The final evaluation results after 20 training epochs were:

Metric	Result
Precision	0.8739
Recall	0.7690
mAP@50	0.8505
mAP@50–95	0.5227

These results are taken from the project’s recorded evaluation run in evaluation/results.csv.

The repository also contains visual evaluation artifacts including:

* Confusion matrix
* Normalised confusion matrix
* Precision curve
* Recall curve
* F1 curve
* Precision–Recall curve
* Training results
* Label distribution visualisation
* Validation prediction examples

⸻

Interactive Inference Pipeline

During inference, the system continuously processes camera frames:

Camera
  ↓
YOLOv10 Detection
  ↓
Detected Object Labels
  ↓
User Speech
  ↓
Speech Recognition
  ↓
Object Relevance Check
  ↓
Question + Object Context
  ↓
Local LLM
  ↓
Generated Answer
  ↓
Text-to-Speech
  ↓
Audio Response

The system also supports follow-up questions about the detected object.

⸻

Local LLM Integration

LM Studio provides a local OpenAI-compatible API endpoint.

The application connects to:

http://localhost:1234/v1

The language model receives a structured prompt containing:

User question
+
Detected object context

For example:

Question:
What is this object used for?
Context:
The detected object is bottle.

The LLM then generates a natural-language response which is converted to speech.

This architecture demonstrates how an object-detection system can be connected to a local language model to create a more interactive AI application.

⸻
## Repository Structure

    real-time-vision-llm-assistant/

    ├── evaluation/

    │   ├── validation_examples/

    │   │   ├── val_batch0_labels.jpg

    │   │   ├── val_batch0_pred.jpg

    │   │   ├── val_batch1_labels.jpg

    │   │   ├── val_batch1_pred.jpg

    │   │   ├── val_batch2_labels.jpg

    │   │   └── val_batch2_pred.jpg

    │   ├── args.yaml

    │   ├── confusion_matrix.png

    │   ├── confusion_matrix_normalized.png

    │   ├── F1_curve.png

    │   ├── labels.jpg

    │   ├── labels_correlogram.jpg

    │   ├── P_curve.png

    │   ├── PR_curve.png

    │   ├── R_curve.png

    │   ├── results.csv

    │   ├── results.png

    │   └── train_batch*.jpg

    ├── src/

    │   └── main.py

    ├── training/

    │   └── train_yolov10.py

    ├── assets/

    ├── experiments/

    ├── .gitignore

    ├── requirements.txt

    └── README.md

Installation

1. Clone the repository

git clone https://github.com/Carlton04/real-time-vision-llm-assistant.git
cd real-time-vision-llm-assistant

2. Create a virtual environment

python -m venv .venv

Activate it on macOS/Linux:

source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

⸻

Model Setup

The inference script expects the YOLOv10 model file:

yolov10m.pt

Place the model weights where the Ultralytics loader can access them, or allow Ultralytics to download the pretrained model when supported by your environment.

The repository does not include .pt model weights because model files are excluded through .gitignore.

⸻

LM Studio Setup

Install and launch LM Studio separately.

Load a compatible local language model and start the local server.

The application expects the OpenAI-compatible endpoint:

http://localhost:1234/v1

The current Python client configuration is:

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

The api_key value shown here is a local placeholder used by the LM Studio-compatible interface; it is not a real secret.

⸻

Running the Assistant

After installing the dependencies, loading the required YOLO model, and starting the LM Studio server:

python src/main.py

The system will:

1. Open the camera.
2. Detect objects using YOLOv10.
3. Display bounding boxes and labels.
4. Announce newly detected objects.
5. Listen for a spoken question.
6. Match the question to a detected object.
7. Send the question and object context to the local LLM.
8. Speak the generated response.
9. Continue accepting follow-up questions.

Press:

q

to exit the camera window.

⸻

Training

The training workflow is located at:

training/train_yolov10.py

The script uses:

model = YOLO("yolov10m.pt")

and trains using:

results = model.train(
    data=data_yaml,
    epochs=20,
    imgsz=640,
    batch=8,
    device=device,
    save=True,
    project="runs",
    name="custom_training",
    workers=2,
    plots=True
)

The SKU-110K dataset and its YAML configuration are not included in this repository.

Before running the training script, the dataset must be obtained separately and the SKU-110K.yaml path/configuration must be available to the training workflow.

Training outputs such as model weights and runs/ are excluded from Git through .gitignore.

⸻

Reproducibility Notes

This repository contains the application source code, training script, evaluation outputs, and documentation.

The following are intentionally excluded:

* Dataset files
* Trained model weights
* Training run directories
* Temporary audio files
* Virtual environments
* IDE-specific files
* API credentials and environment secrets

This keeps the repository lightweight while documenting the main research and engineering workflow.

⸻

Limitations

The current prototype has several limitations:

* Object detection performance depends on the training dataset and environment.
* The application currently uses object labels as language-model context rather than passing raw images to the LLM.
* Speech recognition depends on the configured speech-recognition service and network availability.
* Text-to-speech also requires network access through gTTS.
* The object-to-question matching currently relies on simple text matching.
* Only the detected object labels are retained as conversational visual context.
* The project is primarily a research prototype rather than a production deployment.

⸻

Future Work

Potential improvements include:

* More robust object-to-question grounding
* Temporal tracking of detected objects
* Confidence-aware object selection
* Improved speech recognition and offline speech processing
* Fully offline text-to-speech
* Direct multimodal image-to-language reasoning
* More sophisticated conversational memory
* Object tracking across video frames
* Quantitative latency benchmarking
* Model optimisation and deployment using ONNX or other inference runtimes
* Evaluation across additional datasets and real-world environments

⸻

Skills Demonstrated

This project demonstrates practical experience across:

* Computer Vision
* Object Detection
* YOLOv10
* PyTorch
* OpenCV
* Model Fine-Tuning
* Dataset-Based Evaluation
* Speech Recognition
* Text-to-Speech
* Local LLM Integration
* LM Studio
* OpenAI-Compatible APIs
* Python
* Git/GitHub
* AI System Integration
* Research Prototyping

⸻

Project Context

This project was developed as part of an MSc research project investigating the integration of computer vision, synthetic data/model training, and language-based interaction.

The repository presents the implementation as a standalone research-engineering project, including the inference application, training workflow, evaluation results, and supporting documentation.

⸻

Author

Carlton

GitHub: @Carlton04

⸻

License

No open-source license has currently been specified for this repository.
