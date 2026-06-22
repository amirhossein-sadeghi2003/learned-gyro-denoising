# Learned Gyro Denoising for Attitude Estimation



Open-loop attitude from a raw MEMS gyroscope drifts fast: the small bias and noise on each axis integrate into a growing orientation error. This project trains a small neural network to correct the gyro signal before integration, then checks how much of that drift goes away compared to integrating the raw gyro directly.



The plan is to train and evaluate on a public IMU dataset that ships real ground-truth orientation (EuRoC MAV or TUM-VI), so attitude error is measured against a trusted reference and not against the same sensor. Later I want to run the trained model on my own MPU6050 recordings and check the drift over known rotations as a real-hardware sanity test.



Status: scaffolded. Data loading, the network, and the integration baseline are not implemented yet. The plan lives in docs/project_plan.md.

