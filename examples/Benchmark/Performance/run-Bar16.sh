#!/bin/bash
for d in Vec3f CudaVec3f;
do
let i=1; while [ $i -le 52 ];
do
export s=$i
echo $d - $i - fem
sofaCUDA Bar16-fem-implicit-$d.pscn 1000
mv -f Bar16-fem-implicit-$d.-log.txt Bar16-$i-fem-implicit-$d-log.txt
echo $d - $i - spring
sofaCUDA Bar16-spring-rk4-$d.pscn 1000
mv -f Bar16-spring-rk4-$d.-log.txt Bar16-$i-spring-rk4-$d-log.txt
if [ $i -lt 8 ]; then
let i+=1
else
let i+=4
fi
done
done
