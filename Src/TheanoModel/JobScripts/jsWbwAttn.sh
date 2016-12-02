#!/bin/bash
#PBS -N VQAWBW
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=24:00:00

## SPECIFY JOB NOW
JOBNAME=VQAWBW
CURTIME=$(date +%Y%m%d%H%M%S)
cd $PBS_O_WORKDIR

module load mkl/intel/psxe2015/mklvars
module load lib/hdf/5/1.8.16/gnu
module load compiler/cuda/7.0/compilervars
module load suite/intel/parallelStudio
module load lib/caffedeps/master/intel
module load lib/hdf/4/4.2.11/gnu
module load compiler/python/2.7.10/compilervars
module load apps/Caffe/master/27.01.2016/gnu
module load apps/opencv2.3

WORKDIR=/home/ee/btech/ee1130798/DL/Proj/TheanoModel
PYTHON=/home/ee/btech/ee1130798/anaconda/bin/python

GLOG_log_dir=$WORKDIR THEANO_FLAGS='gcc.cxxflags=-march=core2' $PYTHON $WORKDIR/Code/wbw.py > $WORKDIR/Logs/logWbwAttn.txt 2> $WORKDIR/ErrorLogs/errWbwAttn.txt

