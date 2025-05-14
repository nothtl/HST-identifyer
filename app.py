from fastai.vision.all import *
import gradio as gr
from pathlib import Path

# Load the trained model
MODEL_PATH = Path(__file__).parent / "train_classifier2.pkl"
learn = load_learner(MODEL_PATH)


# Define prediction function
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}


# Create Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="High-Speed Train Classifier",
    description="Upload a photo of a high-speed train from China, Germany, Japan, France or Taiwan. The model will guess which country it comes from.",
)

# Run the web app
if __name__ == "__main__":
    interface.launch()
