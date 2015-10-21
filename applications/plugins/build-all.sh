#! /bin/bash -e

build() {
    pushd $1 >& /dev/null
    pwd
    ~/sofa/scripts/build
    ~/sofa/scripts/install | grep -v -e '-- Up-to-date: '
    popd >& /dev/null
}

# Independant plugins

# build ARTrack                   # Useless shit
build ExternalBehaviorModel
build InvertibleFVM
build MeshSTEPLoader
build MultiThreading
build PluginExample
build ManifoldTopologies
build OptiTrackNatNet
export SIXENSE_ROOT=/home/marc/lair/sixense
build SixenseHydra
# build SofaOpenCL
build SofaSimpleGUI
# build Xitact                    # Windows
# build EmptyCmakePlugin
# build Haption                   # Windows
build ManualMapping
# build PersistentContact         # Doesn't compile
# build SofaPML                   # Useless shit
export OH_SDK_BASE=/home/marc/lair/OpenHaptics/
build Sensable
build SensableEmulation
build SofaCUDA
# build SofaHAPI                  # Windows
build SofaPython
# build SofaQtQuickGUI
build THMPGSpatialHashing
# build Voxelizer                 # Depends on SofaCUDALDI
build SofaCarving

# Beware of dependencies!

build SceneCreator
build SofaTest

build image
build Compliant
# Depends on image
build CGALPlugin
# Depends on image
build Flexible
# Depends on image
build Registration
# Depends on Compliant
# export BULLET_ROOT=~/lair/bullet
# build BulletCollisionDetection
# Depends on Flexible and image
build ColladaSceneLoader
# Depends on Flexible and Compliant
build PreassembledMass
