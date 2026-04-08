# Linear Algebra Image Processing Pipeline

## Overview

This project demonstrates how linear algebra concepts are applied in image processing. The pipeline loads an image, adds noise, removes it using convolution, analyzes the difference, and enhances the final result.

---

## Features

* Load grayscale image as a matrix
* Add Gaussian noise
* Apply Gaussian blur (noise reduction)
* Compute residual (extracted noise)
* Apply sharpening filter
* Visualize all stages

---

## Pipeline

```
Original → Noisy → Cleaned → Residual → Sharpened
```

---

## Core Concepts

### Matrix Representation

An image is treated as a matrix where each element represents pixel intensity.

### Noise Addition

Gaussian noise is added to simulate real-world imperfections:
[
A_{noisy} = A + N
]

### Convolution (Smoothing)

A Gaussian kernel is applied to reduce noise:
[
A_{clean} = K * A_{noisy}
]

### Residual Analysis

Difference between noisy and cleaned image:
[
R = A_{noisy} - A_{clean}
]

### Sharpening

Enhances edges using a sharpening kernel.

---

## Tech Stack

* Python
* NumPy
* OpenCV
* Matplotlib
* SciPy

---

## Project Structure

```
.
├── main.py
├── README.md
└── image.jpg
```

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy opencv-python matplotlib scipy
```

### 2. Update image path

```python
my_image_path = "your_image_path_here"
```

### 3. Run the script

```bash
python main.py
```

---

## Output

The program displays:

* Original image
* Noisy image
* Cleaned (blurred) image
* Residual (noise extracted)
* Final sharpened image

---

## Notes

* If the image path is incorrect, a synthetic image is generated automatically.
* Ensure proper file path for accurate results.

---

## License

For educational purposes.
