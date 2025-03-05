# Face Verification System

This is a **Face Verification System** built using Flask (backend) and HTML/CSS/JavaScript (frontend). It allows users to verify faces by either uploading an image or using their webcam. The system matches the detected face against a reference database and displays the staff details if a match is found.

---

## Features

- **Upload Image**: Users can upload an image for face verification.
- **Webcam Verification**: Real-time face detection and verification using the webcam.
- **Staff Details**: Displays staff details (Staff ID, Batch Number) in a clean, professional card layout.
- **Error Handling**: Displays error messages in a red popup at the top of the screen.
- **Liveliness Check**: Prevents spoofing by checking for subtle changes in the face.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- facenet-pytorch
- A modern web browser (Chrome, Firefox, or Edge)

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/face-verification-system.git
   cd face-verification-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. ## Configuration

- **API Key**: Add your API key in the `.env` file.
- **Reference Images**: Add reference images to the `reference_images` folder.
- **Staff Details**: Update the `staff-details.json` file with your staff information.
   

5. **Run the Application**:
   ```bash
   python app.py
   ```

6. **Access the Application**:
   - Open your browser and navigate to `http://127.0.0.1:5000`.

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

---


## Contact

For questions or feedback, please contact [Vinod Patgar](mailto:vinodpatgar04l@gmail.com).

```



