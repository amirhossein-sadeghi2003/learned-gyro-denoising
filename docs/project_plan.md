# Project plan — learned gyro denoising for attitude estimation

## Goal

Train a neural network to denoise a 3-axis MEMS gyroscope so that open-loop
integration of the corrected signal gives a lower attitude error than
integrating the raw gyro. The network is the centerpiece; the raw-integration
baseline is the thing it has to beat.

## Why

A raw gyro has axis-dependent bias and noise. Integrated open-loop, that turns
into orientation drift that grows with time. Classical fixes (bias subtraction,
complementary or Madgwick filters fused with the accelerometer) work but lean on
extra assumptions. The question here is how far a learned correction on the gyro
alone can push the drift down.

## Data

- Primary: a public IMU dataset with real ground-truth orientation
  (EuRoC MAV or TUM-VI). Ground truth from motion capture / VIO is what makes a
  supervised gyro correction defensible.
- Keep the accelerometer out of the label path so the network is not just
  copying the accelerometer's own attitude estimate.

## Approach

- Input: windows of raw gyro (accel optionally as input features, never as label).
- Model: a small temporal CNN (dilated convolutions) that outputs a per-axis
  gyro correction, in the spirit of Brossard et al., "Denoising IMU Gyroscopes
  with Deep Learning for Open-Loop Attitude Estimation" (2020).
- Integrate the corrected gyro open-loop to orientation.
- Baseline: integrate the raw gyro open-loop with no correction.
- Metric: orientation error vs ground truth over fixed time windows.

## Evaluation

- Compare corrected vs raw drift on held-out sequences.
- Look at where the network helps most (long windows, fast motion, noisy axes).
- Be explicit about what it cannot fix: yaw has no absolute reference without a
  magnetometer.

## Network visualization (a goal of this project)

- Show the learned per-axis correction signal against the raw gyro over time.
- Inspect the dilated-CNN filters and effective receptive field.

## Later (optional capstone)

- Run the trained model on my own MPU6050 recordings.
- Verify against known geometry instead of motion capture: exact 90 deg turns,
  closed-loop return to start, and the reduction in yaw drift.
- A demo on top of the public-dataset core, not the core itself.

## Status

Scaffold only. Nothing in this list is implemented yet.
