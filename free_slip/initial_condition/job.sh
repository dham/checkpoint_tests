#!/bin/bash
#PBS -N Cylinder_IC
#PBS -P xd2
#PBS -q normalbw
#PBS -l walltime=48:00:00 
#PBS -l mem=200GB
#PBS -l ncpus=2
#PBS -l jobfs=10GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#### Load relevant modules:
module use /g/data/xd2/modulefiles
module load firedrake/firedrake-20220301
export OMP_NUM_THREADS=1
#### Now run:
mpirun -mca coll ^hcoll -np $PBS_NCPUS python 2d_cylindrical_IC.py &> output.dat
