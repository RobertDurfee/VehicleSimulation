#!/bin/bash

export GOOGLE_APPLICATION_CREDENTIALS="/afs/athena.mit.edu/user/r/b/rbdurfee/Documents/VehicleSimulation/rl_traffic_intersections_sa.json"

# Preprocess the data
python3 -m sequence.data.preprocess                                            \
  --job-dir gs://vehicle-simulation/jobs/sequence/007                          \
  --data-zip /Users/rbdurfee/Downloads/2017_ford_f150_ecoboost.zip             \
  --in-features Pedal_accel_pos_CAN[per]                                       \
                Brake_pressure_applied_PCM[]                                   \
                Dyno_Spd[mph]                                                  \
  --out-features Dyno_Spd[mph]                                                 \
  --look-back 5                                                                \
  --test-split 0.10                                                            \
  --shuffle

# Start a remote training job with single instance
gcloud ml-engine jobs submit training vehicle_simulation_sequence_007_0        \
  --region us-central1                                                         \
  --scale-tier BASIC_GPU                                                       \
  --runtime-version 1.12                                                       \
  --python-version 3.5                                                         \
  --module-name sequence.trainer.train                                         \
  --package-path sequence/sequence/                                            \
  --job-dir gs://vehicle-simulation/jobs/sequence/007                          \
  --                                                                           \
  --train-file gs://vehicle-simulation/jobs/sequence/007/data/train.csv        \
  --eval-file gs://vehicle-simulation/jobs/sequence/007/data/test.csv          \
  --first-layer-size 200                                                       \
  --num-layers 4                                                               \
  --scale-factor 0.5                                                           \
  --hidden-layer LSTM                                                          \
  --hidden-activation tanh                                                     \
  --output-activation linear                                                   \
  --hidden-dropout 0.4                                                         \
  --optimizer RMSprop                                                          \
  --learning-rate 0.001                                                        \
  --loss mean_squared_error                                                    \
  --train-batch-size 100                                                       \
  --train-epochs 200                                                           \
  --eval-epochs 1                                                              \
  --eval-frequency 3                                                           \
  --checkpoint-frequency 1
