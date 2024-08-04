# YOLO-Number-Plate

This repository contains a complete application framework for number plate detection and recognition. It utilizes a fine-tuned state-of-the-art YOLOv8 model for number plate detection and EasyOCR for text recognition. The app is containerized using Docker.

## Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/yolo-number-plate.git
   cd yolo-number-plate
2. **Install dependencies:**
You can install the required dependencies using requirements.txt.
   ```bash
   pip install -r requirements.txt

## Running the Application

### Using Uvicorn

You can run the application using Uvicorn with the following command:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

### Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t yolo-number-plate .
2. **Run the Docker container:**
   ```bash
   docker run -p 8000:8000 yolo-number-plate

## Contributing

We welcome contributions to enhance the functionality and performance of this project. Please follow these steps:

1. **Fork the repository:**
   Click the "Fork" button at the top right of this page to create a copy of the repository under your own GitHub account.

2. **Clone the forked repository:**
   ```bash
   git clone https://github.com/your-username/yolo-number-plate.git
   cd yolo-number-plate

3. **Create a new branch:**
    ```bash
    git checkout -b feature-branch
4. **Make your changes:**
   Implement your feature or bug fix.
5. **Commit your changes:**
   ```bash
   git commit -m 'Add new feature'
6. **Push to the branch:**
   ```bash
   git push origin feature-branch
7. **Create a pull request:**
   Go to the repository on GitHub and click the "New Pull Request" button. Provide a clear description of your changes.


