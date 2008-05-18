#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H

#include "CudaTypes.h"
#include <sofa/component/MechanicalObject.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class DataTypes>
class CudaKernelsMechanicalObject;

} // namespace cuda

} // namespace gpu

namespace component
{

template<class TCoord, class TDeriv, class TReal>
class MechanicalObjectInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef MechanicalObject<DataTypes> Main;
    typedef typename Main::VecId VecId;
    typedef typename Main::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef gpu::cuda::CudaKernelsMechanicalObject<DataTypes> Kernels;

    /// Temporary storate for dot product operation
    VecDeriv tmpdot;

    static void accumulateForce(Main* m);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, VecId a, VecId b, double f);
    static void vMultiOp(Main* m, const VMultiOp& ops);
    static double vDot(Main* m, VecId a, VecId b);
    static void resetForce(Main* m);
};

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaMechanicalObject_DeclMethods(T) \
    template<> void MechanicalObject< T >::accumulateForce(); \
    template<> void MechanicalObject< T >::vOp(VecId v, VecId a, VecId b, double f); \
    template<> void MechanicalObject< T >::vMultiOp(const VMultiOp& ops); \
    template<> double MechanicalObject< T >::vDot(VecId a, VecId b); \
    template<> void MechanicalObject< T >::resetForce();

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#undef CudaMechanicalObject_DeclMethods

} // namespace component

} // namespace sofa

#endif
