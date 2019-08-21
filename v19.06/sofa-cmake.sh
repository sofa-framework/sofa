#!/bin/bash

# be sure anaconda is not in the path
if [ -d /home/$USER/anaconda2/bin ]; then
  export anac=anaconda2; mv ~/$anac ~/Downloads 
elif [ -d /home/$USER/anaconda3/bin ]; then
	export anac=anaconda3; mv ~/$anac ~/Downloads
fi


cmake -DPLUGIN_IMAGE=ON \
	-DPLUGIN_CGALPLUGIN=ON \
	-DPLUGIN_MANUALMAPPING=ON \
	-DPLUGIN_SOFAALLCOMMONCOMPONENTS=ON \
	-DPLUGIN_SOFADISTANCEGRID=ON \
	-DMODULE_SOFASPARSESOLVER=ON \
	-DPLUGIN_SOFAEULERIANFLUID=ON \
	-DPLUGIN_DIFFUSIONSOLVER=ON \
	-DPLUGIN_SOFAPYTHON=ON \
	-DPLUGIN_VOLUMETRICRENDERING=ON \
	-DSOFA_BUILD_METIS=ON \
	-DPLUGIN_STLIB=ON \
	-DPLUGIN_SOFTROBOTS=ON \
	-DPLUGIN_SOFAROSCONNECTOR=ON \
	-DPLUGIN_ZYROSCONNECTOR=ON \
	-DSOFA_BUILD_MINIFLOWVR=ON \
	-DPLUGIN_MULTITHREADING=ON \
	.. #-DPLUGIN_PSL=ON -DPLUGIN_VolumetricRendering=ON -DPLUGIN_SOFACUDA=ON


echo -e "\nNow making\n\n"

make -j2

echo -e "\n\nNow installing in build folder\n\ns"

make install

if [ -d ~/Downloads/$anac ]; then
	mv ~/Downloads/$anac /home/$USER
fi
