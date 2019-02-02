#!/bin/bash

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\rober\Documents\VehicleSimulation\rl_traffic_intersections_sa.json

# Preprocess the data
python -m sequence.data.preprocess                    ^
  --job-dir gs://vehicle-simulation/jobs/sequence/003 ^
  --data-zip ../data/2017_ford_f150_ecoboost.zip      ^
  --in-features Pedal_accel_pos_CAN[per]              ^
                Brake_pressure_applied_PCM[]          ^
  --out-features Dyno_Spd[mph]                        ^
  --test-split 0.10                                   ^
  --shuffle

# Hyperparameter tuning
gcloud ml-engine jobs submit training vehicle_simulation_sequence_003_0 ^
  --region us-central1                                                  ^
  --scale-tier BASIC                                                    ^
  --runtime-version 1.12                                                ^
  --python-version 3.5                                                  ^
  --module-name sequence.trainer.train                                  ^
  --package-path sequence/sequence/                                     ^
  --job-dir gs://vehicle-simulation/jobs/sequence/003                   ^
  --config ./jobs/sequence/003/tuning.yaml                              ^
  --                                                                    ^
  --train-file gs://vehicle-simulation/jobs/sequence/003/data/train.csv ^
  --eval-file gs://vehicle-simulation/jobs/sequence/003/data/test.csv   ^
  --optimizer RMSprop                                                   ^
  --learning-rate 0.001                                                 ^
  --loss mean_squared_error                                             ^
  --train-batch-size 100                                                ^
  --eval-batch-size 100                                                 ^
  --train-epochs 1                                                      ^
  --eval-epochs 1                                                       ^
  --eval-frequency 1                                                    ^
  --checkpoint-frequency 1
