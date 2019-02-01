#!/bin/bash

python -m data.preprocess                       ^
  --job-dir ./jobs/sequence/001                 ^
  --data-zip ./data/2017_ford_f150_ecoboost.zip ^
  --in-features Pedal_accel_pos_CAN[per]        ^
                Brake_pressure_applied_PCM[]    ^
  --out-features Dyno_Spd[mph]                  ^
  --test-split 0.10                             ^
  --shuffle

gcloud ml-engine local train                      ^
  --module-name sequence.trainer.train            ^
  --package-path sequence/                        ^
  --job-dir ./jobs/sequence/001                   ^
  --                                              ^
  --train-file ./jobs/sequence/001/data/train.csv ^
  --eval-file ./jobs/sequence/001/data/test.csv   ^
  --first-layer-size 200                          ^
  --num-layers 1                                  ^
  --optimizer RMSprop                             ^
  --learning-rate 0.001                           ^
  --loss mean_squared_error                       ^
  --train-batch-size 100                          ^
  --eval-batch-size 100                           ^
  --train-epochs 200                              ^
  --eval-epochs 1                                 ^
  --eval-frequency 1                              ^
  --checkpoint-frequency 1