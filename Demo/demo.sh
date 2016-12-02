CURR_DIR='~/DL/Proj/AttentionVectors' 
scp ee1130798@hpc.iitd.ac.in:$CURR_DIR/$1 ./

python plotAttention.py $1

rm $1