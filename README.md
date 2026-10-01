Real-Time Vision + Local LLM Assistant

A real-time computer vision and voice-interaction system that combines YOLOv10 object detection, speech recognition, and a locally hosted LLM to create an object-grounded conversational assistant.

The system observes objects through a webcam, identifies them using a YOLOv10m detector fine-tuned on the SKU-110K dataset, listens to spoken questions, and uses the detected object as context for a local LLM running through LM Studio.

Demo

▶️ Watch the project demo on YouTube - https://youtu.be/F6J9RCXHqxQ

The demo video is hosted on YouTube rather than stored in this repository because of its large file size.


Overview

This project explores how computer vision, speech interfaces, and local language models can be combined into a single interactive system.

The pipeline is:

                    ┌─────────────────┐
                    │     Webcam      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    YOLOv10m     │
                    │ Object Detection│
                    └────────┬────────┘
                             │
                       Detected objects
                             │
                             ▼
                    ┌─────────────────┐
                    │ Voice Interface │
                    │ Speech-to-Text  │
                    └────────┬────────┘
                             │
                         User query
                             │
                             ▼
                    ┌─────────────────┐
                    │  Context       │
                    │ Object + Query  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Local LLM via   │
                    │    LM Studio    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text-to-Speech  │
                    │     Response    │
                    └─────────────────┘

The language model receives the detected object label as contextual information. The raw camera frame is processed by the object detector and is not directly passed to the LLM in this implementation.

⸻

Key Technical Components

Computer Vision

* YOLOv10m object detector
* Fine-tuning on the SKU-110K retail object detection dataset
* Webcam-based inference
* Bounding-box visualization
* Apple Silicon MPS acceleration when available
* CPU fallback when MPS is unavailable

Voice Interaction

* Speech recognition using SpeechRecognition
* Text-to-speech using gTTS
* Audio playback and speed adjustment using pydub

Local Language Model

* Local LLM inference through LM Studio
* OpenAI-compatible local API interface
* Object detections are supplied as conversational context
* No external LLM API is required for the language-model component during normal local execution

Engineering

* Modular project structure
* Separate training and inference scripts
* Evaluation artifacts retained in the repository
* Training metrics stored in CSV format
* Model weights and datasets excluded from Git using .gitignore

⸻

Model Training

The object detection component uses YOLOv10m, which was fine-tuned on the SKU-110K dataset.

Training configuration used for the reported experiment:

Parameter	Value
Model	YOLOv10m
Dataset	SKU-110K
Epochs	20
Image size	640 × 640
Batch size	8
Device	Apple Silicon MPS when available
Workers	2

The training code is available in:

training/train_yolov10.py

The repository does not include the original dataset or trained .pt model weights. These are intentionally excluded from version control because of their size and dataset/model distribution considerations.

⸻

Evaluation Results

The model was evaluated across 20 training epochs.

Final validation results at epoch 20:

Metric	Result
Precision	0.8739
Recall	0.7690
mAP@50	0.8505
mAP@50–95	0.5227

The training run shows progressive improvement in detection performance throughout the experiment.

For example:

* mAP@50 increased from 0.6408 at epoch 1 to 0.8505 at epoch 20.
* mAP@50–95 increased from 0.3349 to 0.5227.
* Precision increased from 0.6943 to 0.8739.
* Recall increased from 0.5644 to 0.7690.

The complete training history is available in:

evaluation/results.csv

Additional evaluation artifacts include:

evaluation/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── F1_curve.png
├── P_curve.png
├── PR_curve.png
├── R_curve.png
├── results.png
├── labels.jpg
├── labels_correlogram.jpg
└── validation_examples/

These artifacts provide a visual record of the model’s training and validation behaviour.

⸻

Interactive Inference Pipeline

During inference, the system continuously processes webcam frames with YOLOv10.

When an object is detected:

1. The object is displayed with a bounding box.
2. The object label is stored as detected context.
3. The system announces the detected object using text-to-speech.
4. The user can ask a spoken question.
5. Speech is converted into text.
6. The system checks whether the question refers to a detected object.
7. The relevant object label is passed to the local LLM as context.
8. The LLM generates a response.
9. The response is converted back to speech.

This creates a simple vision-to-language interaction loop without requiring a cloud-based language model.

⸻

Local LLM Architecture

The language model is accessed through the OpenAI-compatible API exposed by LM Studio:

Application
     │
     ▼
OpenAI-compatible API
     │
     ▼
LM Studio
     │
     ▼
Local LLM

The current implementation uses:

xtuner/llava-llama-3-8b-v1_1-gguf

The LLM is used for language understanding and response generation, while object detection is handled separately by YOLOv10m.

Therefore, this implementation should be viewed as an object-grounded vision-to-language assistant, rather than an end-to-end multimodal vision-language model.

⸻

Repository Structure

real-time-vision-llm-assistant/
│
├── evaluation/
│   ├── validation_examples/
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
│
├── src/
│   └── main.py
│
├── training/
│   └── train_yolov10.py
│
├── .gitignore
├── requirements.txt
└── README.md

⸻

Installation

Clone the repository:

git clone https://github.com/Carlton04/real-time-vision-llm-assistant.git
cd real-time-vision-llm-assistant

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install the Python dependencies:

pip install -r requirements.txt

On macOS, additional system configuration may be required for microphone/audio dependencies such as PyAudio.

⸻

Model Setup

The inference script expects the YOLOv10m weights:

yolov10m.pt

Place the model weights in the project directory as expected by:

model = YOLO("yolov10m.pt")

The trained custom model weights are intentionally not included in this repository.

⸻

LM Studio Setup

Install and run LM Studio locally, then load a compatible local language model.

The application expects an OpenAI-compatible server running at:

http://localhost:1234/v1

The application connects using:

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

The lm-studio value is a local placeholder used by the LM Studio OpenAI-compatible interface; it is not a cloud API key.

⸻

Running the Assistant

After installing the dependencies, starting the webcam, and launching the LM Studio local server:

python src/main.py

The application will:

1. Open the webcam.
2. Run YOLOv10 object detection.
3. Announce newly detected objects.
4. Listen for spoken questions.
5. Match questions to detected objects.
6. Query the local LLM.
7. Speak the generated response.

Press:

q

to exit the webcam detection loop.

⸻

Training

The training script is located at:

training/train_yolov10.py

The experiment used:

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

The SKU-110K dataset is not included in this repository. To reproduce the training experiment, the dataset must be obtained separately and the dataset configuration path supplied to the training workflow.

Training outputs such as model checkpoints and runs/ directories are excluded from Git.

⸻

Limitations

This project is an experimental research/engineering prototype rather than a production deployment.

Current limitations include:

* The language model operates on object labels rather than raw image features.
* Object-to-question matching currently relies on simple label matching.
* Speech recognition depends on microphone quality and speech recognition availability.
* Text-to-speech uses an external gTTS service.
* The YOLOv10 model weights are not included in the repository.
* The SKU-110K dataset is not included.
* The current system does not provide a formal latency benchmark.
* The system has not been packaged as a production service or deployed to edge hardware.

⸻

Future Work

Potential extensions include:

* More robust semantic matching between spoken questions and detected objects.
* Direct multimodal LLM input using image and language context.
* Object tracking across frames.
* Confidence-aware object selection.
* Quantization and inference optimization for local deployment.
* Formal latency and throughput benchmarking.
* Improved speech recognition and fully local speech processing.
* Containerized deployment.
* Evaluation across additional object detection datasets.
* More systematic ablation studies comparing model sizes and inference configurations.

⸻

Skills Demonstrated

This project demonstrates practical experience across several areas relevant to AI/ML research engineering:

* Computer vision
* Object detection
* Model fine-tuning
* Dataset-driven experimentation
* Evaluation and error analysis
* Local LLM inference
* Vision-to-language system design
* Speech interfaces
* Python
* PyTorch
* Ultralytics
* OpenAI-compatible APIs
* Apple Silicon / MPS acceleration
* Experiment tracking
* Git and GitHub
* Reproducible project organization

⸻

Author

Carlton

GitHub: @Carlton04