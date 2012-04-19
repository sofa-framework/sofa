#/usr/bin/bash
# print all the libs which depend on libcholmod.so.1*

for i in `ls *.so`; do
	echo $i `ldd $i | grep  libcholmod.so.1` | grep libcholmod.so.1
done


