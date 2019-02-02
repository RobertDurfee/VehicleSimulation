#!/bin/bash

python -m sequence.data.preprocess                    ^
  --job-dir gs://vehicle-simulation/jobs/sequence/002 ^
  --data-zip ./data/2017_ford_f150_ecoboost.zip       ^
  --in-features Pedal_accel_pos_CAN[per]              ^
                Brake_pressure_applied_PCM[]          ^
  --out-features Dyno_Spd[mph]                        ^
  --test-split 0.10                                   ^
  --shuffle
