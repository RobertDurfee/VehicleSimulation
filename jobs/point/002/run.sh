#!/bin/bash

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\rober\Documents\VehicleSimulation\rl_traffic_intersections_sa.json

# Preprocess data
python -m point.data.preprocess                    ^
  --job-dir gs://vehicle-simulation/jobs/point/002 ^
  --data-zip ../data/2017_ford_f150_ecoboost.zip   ^
  --in-features Pedal_accel_pos_CAN[per]           ^
                Brake_pressure_applied_PCM[]       ^
  --out-features Dyno_Spd[mph]                     ^
  --test-split 0.10                                ^
  --shuffle

# Remote distributed training job with hypertune
gcloud ml-engine jobs submit training vehicle_simulation_point_002    ^
  --region us-central1                                                ^
  --scale-tier BASIC_GPU                                              ^
  --runtime-version 1.12                                              ^
  --python-version 3.5                                                ^
  --module-name point.trainer.train                                   ^
  --package-path point/point/                                         ^
  --job-dir gs://vehicle-simulation/jobs/point/002                    ^
  --config ./jobs/point/002/tuning.yaml                               ^
  --                                                                  ^
  --train-file gs://vehicle-simulation/jobs/point/002/data/train.csv  ^
  --eval-file gs://vehicle-simulation/jobs/point/002/data/test.csv    ^
  --optimizer Adam                                                    ^
  --learning-rate 0.001                                               ^
  --loss mean_squared_error                                           ^
  --eval-batch-size 32                                                ^
  --eval-epochs 1                                                     ^
  --eval-frequency 5                                                  ^
  --checkpoint-frequency 1