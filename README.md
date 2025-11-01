## 🧠 Brain MRI AI Assistant for Doctors

A sophisticated deep learning and generative AI application designed to assist clinicians. It integrates a Convolutional Neural Network (CNN) for multi-class brain tumor classification, **Grad-CAM** for model explainability, and the **Gemini API** for generating clinical summaries and managing patient-friendly conversations.

-----

## ✨ Key Features & Functionality

  * **🎯 Tumor Classification:** A highly optimized **EfficientNet-B0** model classifies MRI scans into four categories: **Glioma, Meningioma, Pituitary, or No Tumor**.
  * **🔥 Grad-CAM Visualization:** Generates a **Heatmap** overlay (Grad-CAM) to visually highlight the exact regions of the MRI image that the model focused on when making its prediction, enhancing **trust and interpretability**.
  * **🤖 Gemini AI Clinical Summary:** Uses the **Gemini 2.0 Flash** model to generate an immediate, patient-friendly summary report based on the classification result, reported symptoms, and consulting doctor.
  * **💬 Conversational Assistant:** Maintains chat history within the Streamlit session, allowing patients or caregivers to ask follow-up questions about the tumor type, progression, or what questions to ask their doctor.
  * **🖼️ Professional Interface:** Built using **Streamlit** for a clean, interactive, and fast web application experience.

-----

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1\. Prerequisites

  * **Python 3.8+**
  * A **Google Gemini API Key**. You can obtain one from Google AI Studio.
  * The trained model file **`model4.pth`** must be present in the expected directory structure.

### 2\. Project Setup

1.  **Clone the Repository:**

    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <YOUR_REPOSITORY_NAME>
    ```

2.  **Create Model Directory (Required by `app.py`):**
    The application expects the model to be one directory up from `app.py` in a folder named `models`.

    ```bash
    mkdir models
    # Place your trained model file here
    mv path/to/your/model4.pth models/
    ```

3.  **Setup Virtual Environment & Install Dependencies:**

    ```bash
    # Create the environment
    python -m venv venv
    # Activate the environment (on macOS/Linux)
    source venv/bin/activate
    # Install dependencies from the provided list
    pip install -r requirements.txt
    ```

### 3\. API Key Configuration

The application uses Streamlit's built-in secrets management for best practice.

1.  Create a folder named **`.streamlit`** in your project root directory.

2.  Inside the `.streamlit` folder, create a file named **`secrets.toml`** and add your Gemini API Key:

    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
    ```

-----

## 💻 Running the Application

1.  Ensure your virtual environment is active (`source venv/bin/activate`).

2.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3.  The application will automatically launch in your web browser at `http://localhost:8501`.

-----

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | **Main Application File.** Handles the UI, image upload, model prediction, Grad-CAM generation, and all Gemini API interactions. |
| `cnn_brain_tumor_classification.ipynb` | Jupyter Notebook detailing the data preparation, **EfficientNet-B0** training, and final model validation. |
| `models/model4.pth` | **Trained PyTorch Model.** The state dictionary of the final CNN classifier. |
| `requirements.txt` | Explicit list of all required libraries, including `streamlit`, `torch`, `opencv-python-headless`, and `google-genai`. |
| `.streamlit/secrets.toml` | Secure configuration file for the `GEMINI_API_KEY`. |

-----

## 📊 Technical Details & Libraries

### CNN Model Stack

| Library | Version | Purpose |
| :--- | :--- | :--- |
| **`torch` / `torchvision`** | 2.9.0 / 0.24.0 | Core deep learning framework for training and inference. Uses a **pre-trained EfficientNet-B0** architecture. |
| **`opencv-python-headless`** | 4.10.0.84 | Used specifically for the **Grad-CAM** generation, image resizing, and heatmap overlay. |

### Grad-CAM Implementation

The Grad-CAM function works by:

1.  Registering a **forward hook** to capture the feature map activations from the last convolutional layer.
2.  Registering a **backward hook** to capture the gradients flowing back from the prediction score to that same convolutional layer.
3.  Computing the **neuron importance weights** (Global Average Pooling of gradients).
4.  Creating and resizing the final heatmap to overlay on the original MRI image, showing the area of tumor focus.

### Gemini Integration

The **`google-genai`** SDK is used to power the two-stage conversation:

1.  **Initial Summary:** A detailed, context-rich prompt is sent with the prediction, confidence, patient name, doctor, and symptoms to generate a comprehensive, reassuring, and informative first response.
2.  **Follow-up Chat:** A secondary chat loop uses the **`st.session_state`** to maintain conversation history, ensuring context is preserved across follow-up questions from the user.

-----

## 🤝 Contribution

We welcome contributions to improve this clinical tool\! Feel free to fork the repository and submit a Pull Request.

  * Suggesting alternative CNN architectures (e.g., Vision Transformer)
  * Improving the Grad-CAM visualization techniques
  * Refining the Gemini prompts for better clinical accuracy and tone

-----

## 📄 License

This project is licensed under the MIT License.

-----
