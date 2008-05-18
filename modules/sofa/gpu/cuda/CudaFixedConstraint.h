#ifndef SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_H
#define SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_H

#include "CudaTypes.h"
#include <sofa/component/constraint/FixedConstraint.h>

namespace sofa
{

namespace component
{

namespace constraint
{

template<class TCoord, class TDeriv, class TReal>
class FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > Data;
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef FixedConstraint<DataTypes> Main;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename Main::SetIndex SetIndex;

    // min/max fixed indices for contiguous constraints
    int minIndex;
    int maxIndex;
    // vector of indices for general case
    gpu::cuda::CudaVector<int> cudaIndices;

    static void init(Main* m);

    static void addConstraint(Main* m, unsigned int index);

    static void removeConstraint(Main* m, unsigned int index);

    static void projectResponse(Main* m, VecDeriv& dx);
};

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaFixedConstraint_DeclMethods(T) \
    template<> void FixedConstraint< T >::init(); \
    template<> void FixedConstraint< T >::addConstraint(unsigned int index); \
    template<> void FixedConstraint< T >::removeConstraint(unsigned int index); \
    template<> void FixedConstraint< T >::projectResponse(VecDeriv& dx);

CudaFixedConstraint_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaFixedConstraint_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

CudaFixedConstraint_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaFixedConstraint_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#undef CudaFixedConstraint_DeclMethods

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
