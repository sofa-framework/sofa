#!/bin/bash
for d in CudaVec3f;
do
let i=1; while [ $i -le 24 ];
do
export s=$i
echo $d - $i - fem
php examples/Benchmark/Performance/Bar16-fem-implicit-$d.pscn > examples/Benchmark/Performance/Bar16-fem-implicit-$d.scn
sofaCUDA examples/Benchmark/Performance/Bar16-fem-implicit-$d.scn 100
mv -f examples/Benchmark/Performance/Bar16-fem-implicit-$d-log.txt examples/Benchmark/Performance/Bar16-$i-fem-implicit-$d-log.txt
echo $d - $i - spring
php examples/Benchmark/Performance/Bar16-spring-rk4-$d.pscn > examples/Benchmark/Performance/Bar16-spring-rk4-$d.scn
sofaCUDA examples/Benchmark/Performance/Bar16-spring-rk4-$d.scn 100
mv -f examples/Benchmark/Performance/Bar16-spring-rk4-$d-log.txt examples/Benchmark/Performance/Bar16-$i-spring-rk4-$d-log.txt
if [ $i -lt 8 ]; then
let i+=1
else
let i+=4
fi
done
done
