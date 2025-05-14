# HST-Identifier

This is a machine learning project that identifies High-Speed Trains (HSTs) based on their country of origin using image classification. It is built using Fastai, PyTorch, and Gradio.

Users can upload or drag-and-drop an image of a high-speed train, and the model will predict which country it is from (e.g., Japan, China, France).

---

## Demo

Try the app here:  
[https://hst-identifyer-1.onrender.com]


##  How I Developed It

1. **Dataset Collection**:  
   I used DuckDuckGo search to automatically scrape images of high-speed trains from various countries using Fastai's `search_images` function.

2. **Data Cleaning**:  
   I applied Fastai’s `verify_images` to remove corrupted or unreadable files, and resized all images to a uniform resolution using `Resize(224)`.

3. **Model Training**:  
   I used Fastai’s `vision_learner` with a pretrained CNN backbone (`resnet18`) and trained it on the labeled dataset. The DataBlock API was configured with:
   - `get_image_files` to load images
   - `parent_label` to extract labels from folder names
   - `RandomSplitter` to create training and validation sets

4. **Evaluation**:  
   I evaluated performance using accuracy and ensured training/validation loss decreased without overfitting.

5. **Deployment**:  
   I exported the model as a `.pkl` file and built an interactive web UI using Gradio, allowing drag-and-drop image predictions from a browser.

---
