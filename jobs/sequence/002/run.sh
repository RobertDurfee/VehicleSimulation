#!/bin/bash

# Preprocess the data
python -m sequence.data.preprocess                    ^
  --job-dir gs://vehicle-simulation/jobs/sequence/002 ^
  --data-zip ./data/2017_ford_f150_ecoboost.zip       ^
  --in-features Pedal_accel_pos_CAN[per]              ^
                Brake_pressure_applied_PCM[]          ^
  --out-features Dyno_Spd[mph]                        ^
  --test-split 0.10                                   ^
  --shuffle

# Run training locally
gcloud ml-engine local train                                            ^
  --module-name sequence.trainer.train                                  ^
  --package-path sequence/sequence/                                     ^
  --job-dir gs://vehicle-simulation/jobs/sequence/002                   ^
  --                                                                    ^
  --train-file gs://vehicle-simulation/jobs/sequence/002/data/train.csv ^
  --eval-file gs://vehicle-simulation/jobs/sequence/002/data/test.csv   ^
  --first-layer-size 200                                                ^
  --num-layers 1                                                        ^
  --optimizer RMSprop                                                   ^
  --learning-rate 0.001                                                 ^
  --loss mean_squared_error                                             ^
  --train-batch-size 100                                                ^
  --eval-batch-size 100                                                 ^
  --train-epochs 1                                                      ^
  --eval-epochs 1                                                       ^
  --eval-frequency 1                                                    ^
  --checkpoint-frequency 1

# Start a remote training job with single instance
gcloud ml-engine jobs submit training vehicle_simulation_sequence_002   ^
  --region us-central1                                                  ^
  --scale-tier BASIC                                                    ^
  --runtime-version 1.12                                                ^
  --python-version 3.5                                                  ^
  --module-name sequence.trainer.train                                  ^
  --package-path sequence/sequence/                                     ^
  --job-dir gs://vehicle-simulation/jobs/sequence/002                   ^
  --                                                                    ^
  --train-file gs://vehicle-simulation/jobs/sequence/002/data/train.csv ^
  --eval-file gs://vehicle-simulation/jobs/sequence/002/data/test.csv   ^
  --first-layer-size 200                                                ^
  --num-layers 1                                                        ^
  --optimizer RMSprop                                                   ^
  --learning-rate 0.001                                                 ^
  --loss mean_squared_error                                             ^
  --train-batch-size 100                                                ^
  --eval-batch-size 100                                                 ^
  --train-epochs 1                                                      ^
  --eval-epochs 1                                                       ^
  --eval-frequency 1                                                    ^
  --checkpoint-frequency 1
