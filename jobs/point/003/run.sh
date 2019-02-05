#!/bin/bash

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\rober\Documents\VehicleSimulation\rl_traffic_intersections_sa.json

# Preprocess data
python -m point.data.preprocess                    ^
  --job-dir gs://vehicle-simulation/jobs/point/003 ^
  --data-zip ../data/2017_ford_f150_ecoboost.zip   ^
  --in-features Pedal_accel_pos_CAN[per]           ^
                Brake_pressure_applied_PCM[]       ^
  --out-features Dyno_Spd[mph]                     ^
  --test-split 0.10                                ^
  --shuffle

# Remote distributed training job with hypertune
gcloud ml-engine jobs submit training vehicle_simulation_point_003    ^
  --region us-central1                                                ^
  --config ./jobs/point/003/config.yaml                               ^
  --runtime-version 1.12                                              ^
  --python-version 3.5                                                ^
  --module-name point.trainer.train                                   ^
  --package-path point/point/                                         ^
  --job-dir gs://vehicle-simulation/jobs/point/003                    ^
  --                                                                  ^
  --train-file gs://vehicle-simulation/jobs/point/003/data/train.csv  ^
  --eval-file gs://vehicle-simulation/jobs/point/003/data/test.csv    ^
  --first-layer-size 256                                              ^
  --num-layers 4                                                      ^
  --scale-factor 0.5                                                  ^
  --hidden-activation relu                                            ^
  --output-activation linear                                          ^
  --input-dropout 0.0                                                 ^
  --hidden-dropout 0.25                                               ^
  --optimizer Adam                                                    ^
  --learning-rate 0.001                                               ^
  --loss mean_squared_error                                           ^
  --train-batch-size 128                                              ^
  --eval-batch-size 32                                                ^
  --train-epochs 50                                                   ^
  --eval-epochs 1                                                     ^
  --eval-frequency 5                                                  ^
  --checkpoint-frequency 1