#!/bin/bash

set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\rober\Documents\VehicleSimulation\rl_traffic_intersections_sa.json

# Preprocess data
python -m point.data.preprocess                    ^
  --job-dir gs://vehicle-simulation/jobs/point/001 ^
  --data-zip ../data/2017_ford_f150_ecoboost.zip   ^
  --in-features Pedal_accel_pos_CAN[per]           ^
                Brake_pressure_applied_PCM[]       ^
  --out-features Dyno_Spd[mph]                     ^
  --test-split 0.10                                ^
  --shuffle

# Local training
gcloud ml-engine local train                                         ^
  --module-name point.trainer.train                                  ^
  --package-path point/point/                                        ^
  --job-dir gs://vehicle-simulation/jobs/point/001                   ^
  --                                                                 ^
  --train-file gs://vehicle-simulation/jobs/point/001/data/train.csv ^
  --eval-file gs://vehicle-simulation/jobs/point/001/data/test.csv   ^
  --first-layer-size 256                                             ^
  --num-layers 1                                                     ^
  --scale-factor 0.25                                                ^
  --hidden-activation relu                                           ^
  --output-activation linear                                         ^
  --input-dropout 0.0                                                ^
  --hidden-dropout 0.5                                               ^
  --optimizer Adam                                                   ^
  --learning-rate 0.001                                              ^
  --loss mean_squared_error                                          ^
  --train-batch-size 32                                              ^
  --eval-batch-size 32                                               ^
  --train-epochs 1                                                   ^
  --eval-epochs 1                                                    ^
  --eval-frequency 1                                                 ^
  --checkpoint-frequency 1
