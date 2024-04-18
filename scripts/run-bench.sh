#!/usr/bin/bash


export OMP_NUM_THREADS=64
exec 4<"mtx-paths.txt"

while read -u4 MTXPATH; do
    
    echo $MTXPATH

    
    MTX=$(echo $MTXPATH | grep -oP '/\K\S+')

    if grep -wq "${MTX}" log.out; then
        echo "${MTX} already executed"
        continue
    fi

    wget "https://suitesparse-collection-website.herokuapp.com/MM/"$MTXPATH".tar.gz"
    tar -xzf $MTX".tar.gz"

    rm -f $MTX".tar.gz"

    mpirun -n 1 ../build/spgemm-bench $MTX/$MTX.mtx $MTX/$MTX.mtx 0
    mpirun -n 1 ../build/spgemm-bench $MTX/$MTX.mtx $MTX/$MTX.mtx 1

    rm -rf $MTX

    echo $MTX >> log.out

    mv $MTXx$MTX* ../data

done




