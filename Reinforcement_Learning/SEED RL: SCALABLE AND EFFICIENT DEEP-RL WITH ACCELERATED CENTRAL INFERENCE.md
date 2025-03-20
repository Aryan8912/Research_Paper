# SEED RL: SCALABLE AND EFFICIENT DEEP-RL WITH ACCELERATED CENTRAL INFERENCE

 IMPALA architecture which is used in various forms in Ape-X, OpenAI Rapid and others
 
![Screenshot (209)](https://github.com/user-attachments/assets/81dbe82f-88ae-4ec1-b749-f7e31898ae62)


![Screenshot (210)](https://github.com/user-attachments/assets/9522e7c7-911d-475c-b713-8e175cb68f07)

![Screenshot (211)](https://github.com/user-attachments/assets/fdf23422-e09e-40f3-8152-6b7d852243c8)

Key Components of the SEED Learner Architecture

The diagram is divided into different modules:

Inference Threads (Left Side)

Recurrent States Storage: Stores internal states for recurrent policies.

Incomplete Trajectories Store: Stores partial episodes that haven't been completed yet.

Complete Trajectories Queue: Holds full trajectories (state-action-reward sequences) that can be used for training.

These components interact with the batching layer to efficiently prepare training data.

Training Thread (Right Side)

Device Buffer: Temporarily holds data before sending it to the model for gradient updates.

Prioritized Replay Buffer (Optional): If used, it stores and samples trajectories based on their importance (prioritization).

Training TPUs/GPUs: The model parameters are updated using backpropagation by applying gradients.

Model (Center)

The core policy model is responsible for both inference and learning.

It is used by inference TPUs to generate actions in real-time and by training TPUs to update parameters.

Data Flow in SEED

Step 1: Complete trajectories are stored in the prioritized replay buffer or directly used for training.

Step 2: Data from the buffer is sent to the model for training.

Step 3: Updated gradients are applied to optimize the model.

The process continues iteratively, ensuring that the model is trained asynchronously while inference continues.

Reporting and Checkpointing

The learner periodically saves checkpoints and logs performance metrics.

a trajectory is fully unrolled it is added to a FIFO queue or replay buffer and later sampled by data prefetching threads. Finally, the trajectories are pushed to a device buffer for each of the TPU cores taking part in training. The training thread (the main Python thread) takes the prefetched trajectories, computes gradients using the training TPU cores and applies the gradients on the models of all TPU cores (inference and training) synchronously. The ratio of inference and training cores can be adjusted for maximum throughput and utilization. The architecture scales to a TPU pod (2048 cores) by roundrobin assigning actors to TPU host machines, and having separate inference threads for each TPU host. When actors wait for a response from the learner, they are idle so in order to fully utilize the machines, we run multiple environments on a single actor.


Understanding "Near On-Policy" Training

Unlike traditional on-policy RL (e.g., A2C), where the agent is trained on the latest data only, SEED maintains a near on-policy approach by efficiently using past trajectories.

It achieves this by asynchronously collecting experience and training, sometimes incorporating replay buffers to stabilize learning.

![Screenshot (212)](https://github.com/user-attachments/assets/ac739727-465c-47e0-978b-59dc7207b51d)

![Screenshot (216)](https://github.com/user-attachments/assets/8dc165c3-9df8-4af1-9144-5d55b2256205)


![Screenshot (215)](https://github.com/user-attachments/assets/ecff7768-aa63-4195-959d-ea471b57b6cf)


![Screenshot (214)](https://github.com/user-attachments/assets/d05d3e57-e93e-43de-a9ce-ee949ffaf567)


![Screenshot (213)](https://github.com/user-attachments/assets/f2d434dd-5ab8-40c5-9f9b-886aa1ed04fc)

