## Introduction

SofaCUDA is a plugin that provides a number of Sofa components that have been implemented using CUDA, which allows the graphics card to be used for multithreaded programming. This can significantly improve the performance of your scene when working with large meshes.

## Compilation

See [instructions on the SOFA website](https://www.sofa-framework.org/community/doc/plugins/usual-plugins/using-cuda/).

## Paradigm

The plugin is mainly a list of components that already exist in various modules of SOFA.
The difference is that those components are now implemented using CUDA.

In SOFA, components are defined to work with specific data types, such as `Vec3Types` or `Rigid3Types` (most of them are defined in `VecTypes.h` and `RigidTypes.h`).
SofaCUDA adds more data types.
They still represent the same type of data, but they are implemented such that they can be manipulated on the GPU using CUDA.
For example, the CPU data type `Vec3Types` equivalent in SofaCUDA is `CudaVec3Types`.
CUDA data types are defined in `CudaTypes.h`.

In SOFA, components working with a specific data type are just template classes explicitly instantiated with this data type.
For example, `MechanicalObject<Vec3Types>` is the instantiation of the template class `MechanicalObject` with `Vec3Types` as the template parameter.
In SofaCUDA, components are template specializations of the SOFA equivalents.
For example, `MechanicalObject<CudaVec3Types>` is a specialization of `MechanicalObject` for `CudaVec3Types`.

The result is that the user just has to change the template parameter in the scene, in order to use the CUDA version of the same component.
For example, in XML, `<MechanicalObject template="Vec3f" />` is the CPU version of `MechanicalObject`, and `<MechanicalObject template="CudaVec3f" />` is the GPU version.
Note that in SOFA Vec3Types are by default in double (Vec3d), except if you compile in float only (CMake option: SOFA_FLOATING_POINT_TYPE). However CudaVec3Types are using float on the device. You can activate double precision using CMake option: SOFACUDA_DOUBLE).

## Performances

The folder `scenes/benchmarks` contains simulation files used to compare performances between CPU and GPU implementations.
All scenes exist in double: the CPU version has the suffix `_cpu`, and the GPU version has the suffix `_gpu`.

To give you an idea of the speed up you can achieve using SofaCUDA, measurement have been performed on some systems on some simulations in `scenes/benchmarks`.
It is provided as a CSV file in the `doc` folder.
Here is the corresponding graph:
![performances_csv](https://user-images.githubusercontent.com/21199245/148964875-99c21320-384f-40c9-9dc6-a5e77a8dca82.png)