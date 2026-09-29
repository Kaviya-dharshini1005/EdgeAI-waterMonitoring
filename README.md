# EdgeAI_WaterGuard

## RTOS-Based Adaptive Water Quality Monitoring and Edge AI Anomaly Detection System

EdgeAI WaterGuard is an embedded water-quality monitoring system developed using the STM32H533RE ARM Cortex-M33. The system combines pH, TDS, and turbidity sensing with Edge AI and RTOS-based task management to identify abnormal water-quality conditions.

The main idea is to perform the important analysis locally on the microcontroller instead of depending on continuous cloud connectivity. A compact autoencoder is trained to learn the characteristics of normal water-quality data. During operation, new sensor measurements are passed through the trained model, and the reconstruction error is used to identify unusual combinations of water-quality parameters.

The project also uses slope-based trend detection, confidence-based anomaly confirmation, and dynamic RTOS task-priority management to make the system more robust to sensor noise and changing conditions.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [How the System Works](#how-the-system-works)
- [Water Quality Parameters](#water-quality-parameters)
- [Data Preprocessing](#data-preprocessing)
- [Autoencoder Model](#autoencoder-model)
- [How the Neural Network Works](#how-the-neural-network-works)
- [Model Training](#model-training)
- [Reconstruction Error](#reconstruction-error)
- [Anomaly Detection](#anomaly-detection)
- [Slope-Based Detection](#slope-based-detection)
- [Confidence-Based Detection](#confidence-based-detection)
- [RTOS Implementation](#rtos-implementation)
- [Dynamic Priority Scheduling](#dynamic-priority-scheduling)
- [Embedded Deployment](#embedded-deployment)
- [Complete Workflow](#complete-workflow)
- [Key Features and Novelty](#key-features-and-novelty)
- [Hardware](#hardware)
- [Software and Tools](#software-and-tools)
- [Project Development](#project-development)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Current Status](#current-status)
- [Future Work](#future-work)
- [Applications](#applications)

---

## Overview

Water quality is affected by several parameters at the same time. Monitoring each parameter independently using fixed thresholds may not always identify abnormal conditions effectively.

EdgeAI WaterGuard uses three primary parameters:

- pH
- TDS
- Turbidity

These measurements are processed together rather than being treated as completely independent values.

The project uses an autoencoder because it can learn the pattern of normal water-quality measurements without requiring every possible type of contamination to be explicitly labelled.

When a new measurement does not resemble the patterns learned during training, the model can produce a larger reconstruction error. This error is then used as one of the indicators for anomaly detection.

---

## Problem Statement

Conventional water-quality monitoring systems often depend on manually selected thresholds.

For example, a system may define separate limits for pH, TDS, and turbidity and generate an alert whenever one of the limits is crossed.

This approach has several limitations.

A water-quality condition can depend on the combination of multiple parameters rather than one parameter alone. Sensor readings can also contain noise, and a single abnormal reading does not necessarily mean that the water condition has actually changed.

The project therefore aims to develop a system that can:

1. Monitor multiple water-quality parameters.
2. Learn normal water-quality patterns using Edge AI.
3. Detect unusual combinations of sensor values.
4. Consider changes in sensor values over time.
5. Reduce false alarms caused by temporary sensor noise.
6. Adapt RTOS task priorities according to the detected system condition.
7. Perform the core AI inference locally on the microcontroller.

---

## Proposed Solution

EdgeAI WaterGuard combines sensor monitoring, machine learning, and real-time operating-system functionality.

The pH, TDS, and turbidity sensors provide the input measurements. These values are acquired by the STM32H533RE and normalized using the same parameters used during model training.

The normalized values are then passed to a compact autoencoder.

The autoencoder attempts to reconstruct the original input. The difference between the original and reconstructed values is measured using Mean Squared Error.

If the reconstruction error becomes sufficiently high, the reading is treated as a potential anomaly.

The system does not depend only on this single value. It can also examine the trend of sensor readings using slope-based detection and use a confidence counter so that one noisy reading does not immediately change the system state.

The resulting system condition can then influence the priority of important RTOS tasks.

---

## How the System Works

The system operates through the following stages:

1. The water-quality sensors provide pH, TDS, and turbidity measurements.
2. The STM32 acquires the sensor values.
3. The raw values are converted into the required measurement format.
4. The measurements are normalized using the training-time normalization parameters.
5. The normalized values are provided to the embedded autoencoder.
6. The autoencoder reconstructs the three input values.
7. Reconstruction error is calculated.
8. The error is compared with the anomaly threshold.
9. Sensor trends can be evaluated using slope-based detection.
10. A confidence mechanism checks whether the abnormal condition persists.
11. The system determines whether the current condition is normal or anomalous.
12. The RTOS can adjust task priorities according to the detected condition.
13. The system continues monitoring the next set of sensor readings.

---

## Water Quality Parameters

### pH

pH represents the acidity or alkalinity of water.

The pH value is one of the three inputs provided to the machine-learning model.

### TDS

Total Dissolved Solids represents the concentration of dissolved substances in water.

TDS provides information that is different from pH and turbidity, making it useful as part of the combined input.

### Turbidity

Turbidity represents the cloudiness of water caused by suspended particles.

It provides another independent measurement of water condition.

The three parameters are considered together because the objective is to identify unusual combinations rather than simply checking each parameter separately.

---

## Data Preprocessing

The three parameters have very different numerical ranges.

For example, pH is normally represented on a relatively small scale, while TDS can contain values in the hundreds or thousands.

If the raw values were directly given to the neural network, the larger numerical values could dominate the calculations.

Therefore, the training pipeline uses Min-Max normalization.

The normalized value is calculated as:

$$
x_{norm} =
\frac{x-x_{min}}
{x_{max}-x_{min}}
$$

where:

- $x$ is the original value.
- $x_{min}$ is the minimum training value.
- $x_{max}$ is the maximum training value.
- $x_{norm}$ is the normalized value.

The same normalization parameters used during training are stored for embedded inference.

This is important because real sensor values must be processed in exactly the same way as the values used to train the model.

---

## Autoencoder Model

The machine-learning component is a compact autoencoder.

The current architecture is:

**3 → 8 → 4 → 2 → 4 → 8 → 3**

The three input neurons correspond to:

- pH
- TDS
- Turbidity

The encoder gradually reduces the information into a two-neuron latent representation.

The decoder then expands the representation back into three output values.

The structure can therefore be understood as:

**Input → Encoder → Latent Space → Decoder → Reconstruction**

The two-neuron middle layer is the bottleneck of the model.

The purpose of this bottleneck is to force the network to learn a compact representation of the normal relationship between the three water-quality parameters.

---

## How the Neural Network Works

Each neuron performs a weighted-sum calculation followed by an activation function.

The basic calculation is:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

where:

- $x_i$ is an input value.
- $w_i$ is the corresponding weight.
- $b$ is the bias.
- $z$ is the weighted sum.

For the hidden layers, ReLU activation is used:

$$
ReLU(z)=max(0,z)
$$

Therefore, a hidden layer can be represented as:

$$
a = ReLU(Wx+b)
$$

The output layer reconstructs the three original features.

During training, the network learns the weights and biases that allow it to reconstruct normal water-quality patterns accurately.

---

## Model Training

The model is trained offline using Python and TensorFlow.

The training process consists of:

1. Loading the water-quality dataset.
2. Cleaning the dataset.
3. Selecting the required features.
4. Selecting normal water samples for training.
5. Normalizing the data.
6. Creating the autoencoder.
7. Training the network.
8. Monitoring the training loss.
9. Calculating reconstruction errors.
10. Determining an anomaly threshold.
11. Saving the trained model.
12. Extracting the required model parameters for embedded deployment.

The model is trained to learn normal behaviour rather than simply memorizing a list of contaminated-water examples.

This is important for anomaly detection because the objective is to identify data that differs significantly from the learned normal pattern.

---

## Reconstruction Error

After training, a new normalized input is passed through the autoencoder.

The model produces reconstructed values.

For example:

$$
x = [pH,\ TDS,\ Turbidity]
$$

The autoencoder produces:

$$
\hat{x} = [\hat{pH},\ \hat{TDS},\ \hat{Turbidity}]
$$

The difference between the input and reconstructed output is calculated using Mean Squared Error.

$$
MSE =
\frac{1}{3}
\sum_{i=1}^{3}
(x_i-\hat{x}_i)^2
$$

A smaller reconstruction error means that the model reconstructed the input more closely.

A larger reconstruction error indicates that the input differs more significantly from the patterns learned by the model.

---

## Anomaly Detection

The reconstruction error is compared with an anomaly threshold.

Conceptually:

**Low reconstruction error → likely normal pattern**

**High reconstruction error → potentially abnormal pattern**

The threshold is determined during the model-development stage and stored for use by the embedded system.

The anomaly detector therefore does not require TensorFlow to be running on the STM32.

The microcontroller only needs the trained parameters, normalization values, and anomaly threshold required for inference.

---

## Slope-Based Detection

The system also considers how sensor values change over time.

A single sensor reading can be affected by noise or temporary fluctuations.

For this reason, the project includes slope-based analysis as an additional detection mechanism.

A simple slope can be calculated as:

$$
Slope =
\frac{y_2-y_1}
{t_2-t_1}
$$

If a parameter continues to move in the same direction over multiple readings, this can provide evidence of a developing condition.

Slope-based analysis therefore complements the autoencoder:

- The autoencoder looks at the relationship between multiple parameters.
- Slope analysis looks at how the values change over time.

This gives the system both pattern-based and trend-based information.

---

## Confidence-Based Detection

Sensor readings are not perfectly stable.

A single noisy reading should not immediately cause the entire system to change its state.

A confidence counter is therefore used to confirm persistent anomalies.

For example, if the reconstruction error exceeds the threshold:

- First abnormal evaluation → confidence increases.
- Second consecutive abnormal evaluation → confidence increases again.
- Third consecutive abnormal evaluation → anomaly can be confirmed.

If the next reading returns to normal, the confidence state can be reduced or reset depending on the implemented logic.

This approach reduces false positives caused by short-duration sensor noise.

---

## RTOS Implementation

The embedded system is designed using μT-Kernel 3.0 RTOS.

Instead of placing all operations inside one sequential program loop, the system can divide the work into independent tasks.

The main logical operations include:

### Sensor Acquisition

Responsible for periodically obtaining the pH, TDS, and turbidity values.

### AI Processing

Responsible for normalization, neural-network calculations, and reconstruction.

### Anomaly Decision

Responsible for reconstruction-error evaluation, threshold comparison, confidence handling, and trend analysis.

### Communication and Output

Responsible for sending or displaying the system state and handling required outputs.

### System Management

Responsible for overall monitoring and coordination of the application.

The RTOS provides scheduling and synchronization between these operations.

---

## Dynamic Priority Scheduling

A major part of the project is the use of dynamic task-priority adjustment.

In a conventional RTOS application, tasks normally operate with fixed priorities.

In EdgeAI WaterGuard, the detected system condition can influence the priority of important tasks.

During normal operation, the system can continue with its normal task-priority configuration.

When an anomaly is confirmed, critical processing tasks can receive higher priority so that the system responds more quickly to the abnormal condition.

The μT-Kernel priority-management functionality is therefore used as part of the adaptive behaviour of the monitoring system.

This connects the machine-learning decision with the real-time scheduling behaviour of the embedded system.

---

## Embedded Deployment

The machine-learning model is trained on a computer because model training requires more computational resources than the target microcontroller is intended to provide.

The deployment therefore has two stages.

### Offline Stage

Python and TensorFlow are used to:

- Prepare the dataset.
- Normalize the data.
- Train the autoencoder.
- Evaluate reconstruction error.
- Determine the anomaly threshold.
- Extract weights and biases.

### Embedded Stage

The trained parameters are converted into C-compatible data and included in the STM32 project.

The STM32 then performs the forward-pass calculations using:

- Sensor values
- Normalization parameters
- Neural-network weights
- Neural-network biases
- Anomaly threshold

TensorFlow itself is not required to run on the STM32.

The board performs the required mathematical operations directly in embedded C.

---

## Complete Workflow

The complete project workflow starts with the water-quality sensors.

The sensors provide pH, TDS, and turbidity measurements to the STM32H533RE.

The STM32 preprocesses and normalizes these values and passes them to the embedded autoencoder.

The autoencoder reconstructs the input and the system calculates the reconstruction error.

The error is compared with the anomaly threshold. At the same time, the system can examine the trend of sensor values using slope analysis.

A confidence mechanism prevents a single noisy reading from immediately being treated as a confirmed anomaly.

Once the system determines the current condition, the RTOS can adjust task priorities so that critical processing receives appropriate CPU attention.

The system then continues monitoring the next sensor sample.

---

## Key Features and Novelty

### Autoencoder-Based Anomaly Detection

The system learns normal water-quality patterns and uses reconstruction error to identify unusual measurements.

### Multi-Parameter Analysis

pH, TDS, and turbidity are processed together rather than relying entirely on independent threshold checks.

### Edge AI

The trained model is deployed locally on the STM32 so that the core anomaly-detection process can run without continuous cloud processing.

### Slope-Based Detection

The system can detect persistent changes in sensor values instead of reacting only to individual readings.

### Confidence-Based Anomaly Confirmation

Multiple abnormal evaluations can be required before the system confirms an anomaly, reducing false positives caused by temporary noise.

### Dynamic RTOS Priority Scheduling

The detected water-quality condition can influence RTOS task priorities, allowing the system to adapt its processing behaviour.

### Compact Neural Network

The 3 → 8 → 4 → 2 → 4 → 8 → 3 model is designed to keep the embedded inference computationally manageable while still providing a latent representation for anomaly detection.

---

## Hardware

The main hardware platform is:

- STM32H533RE
- ARM Cortex-M33
- pH sensor
- TDS sensor
- Turbidity sensor
- Required ADC/signal interfaces
- Output or alert interface

The STM32H533RE acts as the central processing platform for sensor acquisition, AI inference, anomaly detection, and RTOS scheduling.

---

## Software and Tools

### Machine Learning

- Python
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Embedded Development

- STM32CubeIDE
- Embedded C
- STM32H533RE
- ARM Cortex-M33

### RTOS

- μT-Kernel 3.0

---

## Project Development

The project was developed in stages.

### Stage 1 — Problem Definition

The water-quality monitoring problem was defined and pH, TDS, and turbidity were selected as the primary parameters.

### Stage 2 — Dataset Preparation

The available water-quality dataset was cleaned and the required features were selected.

Normal samples were used to train the autoencoder.

### Stage 3 — Autoencoder Development

The autoencoder architecture was developed using TensorFlow.

The final architecture uses:

**3 → 8 → 4 → 2 → 4 → 8 → 3**

### Stage 4 — Model Training

The model was trained using normalized water-quality data.

Training loss and reconstruction errors were evaluated during development.

### Stage 5 — Threshold Development

Reconstruction errors were analyzed and an anomaly threshold was generated for embedded use.

### Stage 6 — Embedded Conversion

The trained weights, biases, normalization values, and threshold were converted into C-compatible data for deployment on the STM32.

### Stage 7 — RTOS Integration

The embedded application was structured around μT-Kernel 3.0 tasks.

The system design also includes confidence handling, slope-based analysis, and dynamic priority management.

### Stage 8 — Hardware Validation

The final stage involves testing the complete system with real sensor readings and evaluating its response under different water conditions.

---

## Results

The machine-learning development stage produced:

- A trained TensorFlow autoencoder.
- Training-loss data.
- Reconstruction-error data.
- Normalization parameters.
- Neural-network weights and biases.
- Anomaly threshold values.
- C-compatible model parameters for embedded integration.

The embedded stage focuses on reproducing the trained model's inference behaviour on the STM32H533RE and validating its performance with real sensor measurements.

Performance evaluation includes inference time, memory usage, scheduling behaviour, anomaly-detection consistency, and sensor noise response.

---
## Demo Video
https://www.youtube.com/watch?v=ABC123xyz

## Repository Structure

The repository contains the machine-learning code, generated model parameters, embedded implementation, and supporting project files.

A typical structure is:

```text
EdgeAI_WaterGuard/
├── Dataset/
├── ML/
├── Model/
├── STM32/
├── outputs/
├── train_autoencoder_tf.py
├── requirements.txt
└── README.md


