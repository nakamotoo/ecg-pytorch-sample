#!/bin/bash
for outcome in MR
    do for optim in adam
        do for nhead in 2 4
            do for nlayers in 1 3
                do
                echo $outcome $optim $nhead $nlayers
                python run_tf.py Model_TFEncoder_AttnMap $optim $outcome $nhead $nlayers
                python run_tf.py Model_TFEncoder_AttnMap_V2 $optim $outcome $nhead $nlayers
                done
            done
        done
    done