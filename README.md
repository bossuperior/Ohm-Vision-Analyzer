# Ohm Vision Analyzer

**Intelligent Breadboard Topology Extraction and Resistance Analysis System**

## Overview
Ohm Vision Analyzer is a Senior Engineering Project designed to automate the analysis of electrical circuits constructed on breadboards. By integrating computer vision and graph theory, the system extracts circuit topology from a single input image, identifies electronic components (resistors and jumper wires), and computes the total equivalent resistance (Rtotal) of the circuit network.

---

## System Pipeline
The system utilizes a modular processing pipeline to ensure high accuracy and modularity:

```text
ArUco Tag Detection 
    |
    v
Perspective Warp (Rectification)
    |
    v
Pose Estimation (Component Localization)
    |
    v
Leg-to-Body Crop & Align
    |
    v
Resistance Classification (Color Band Identification)
    |
    v
Grid Mapper -> NetworkX (Circuit Analysis & Results)
```

---

## Key Features

* ** Automated Geometric Rectification: Implements ArUco marker detection to perform perspective transformation, normalizing the breadboard view regardless of camera angle.
* ** Component Localization:**  Uses pose estimation models to precisely locate resistor bodies and wire connection points.
* ** Automated Component Identification:**  Employs deep learning-based classification to read resistance values from color bands, ensuring robust detection even in variable lighting conditions.
* ** Circuit Topology Reconstruction:**  Maps visual data to a grid-based graph representation, enabling topological analysis through NetworkX.
* ** Graph-Theoretic Analysis:**  Calculates equivalent resistance and interprets circuit configurations (series, parallel, mixed, Wheatstone bridge) using nodal admittance matrix methods.
* ** Modular Design:**  A decoupled architecture allows for independent benchmarking of detection and classification models.

---

## Technical Stack
* ** Computer Vision:**  OpenCV
* ** Pose Estimation and Classification Models.** 
* ** Neural Network Frameworks:**  PyTorch, MMPose (RTMPose, HigherHRNet).
* ** Circuit & Graph Analysis:**  NetworkX, NumPy, SciPy.
* ** Development:**  Python 3.10+, PEP 8 compliant modular structure.

---

## Project Structure
```text
src/vision/: Image preprocessing, ArUco detection, and perspective warping modules.

src/inference/: Model execution engines for pose estimation and classification.

src/topology/: Graph construction, grid mapping, and circuit analysis logic using NetworkX.

src/ui/: GUI implementation for result visualization.

config/: Configuration files and hyperparameters.
```

## Contributors
* ** 6652300371 Chaloemphon Bannarak ** 
* ** 6652300931 Komkrit Tungtatiyapat ** 