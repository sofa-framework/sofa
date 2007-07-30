#ifndef SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/forcefield/MeshSpringForceField.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

template <>
class SpringForceFieldInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    //enum { BSIZE=16 };
    struct GPUSpring
    {
        int index; ///< -1 if no spring
        float initpos;
        float ks;
        float kd;
        GPUSpring() : index(-1), initpos(0), ks(0), kd(0) {}
        void set(int index, float initpos, float ks, float kd)
        {
            this->index = index;
            this->initpos = initpos;
            this->ks = ks;
            this->kd = kd;
        }
    };
    struct GPUSpringSet
    {
        int vertex0; ///< index of the first vertex connected to a spring
        int nbVertex; ///< number of vertices to process to compute all springs
        int nbSpringPerVertex; ///< max number of springs connected to a vertex
        gpu::cuda::CudaVector<GPUSpring> springs; ///< springs attached to each points (layout per bloc of NBLOC vertices, with first spring of each vertex, then second spring, etc)
        gpu::cuda::CudaVector<float> dfdx; ///< only used for StiffSpringForceField
        GPUSpringSet() : vertex0(0), nbVertex(0), nbSpringPerVertex(0) {}
        void init(int v0, int nbv, int nbsperv)
        {
            vertex0 = v0;
            nbVertex = nbv;
            nbSpringPerVertex = nbsperv;
            int nbloc = (nbVertex+BSIZE-1)/BSIZE;
            springs.resize(nbloc*nbSpringPerVertex*BSIZE);
        }
        void set(int vertex, int spring, int index, float initpos, float ks, float kd)
        {
            int bloc = vertex/BSIZE;
            int b_x  = vertex%BSIZE;
            springs[ bloc*BSIZE*nbSpringPerVertex // start of the bloc
                    + spring*BSIZE                 // offset to the spring
                    + b_x                          // offset to the vertex
                   ].set(index, initpos, ks, kd);
        }
    };
    GPUSpringSet springs1; ///< springs from model1 to model2
    GPUSpringSet springs2; ///< springs from model2 to model1 (only used if model1 != model2)
};

//
// SpringForceField
//

template <>
void SpringForceField<gpu::cuda::CudaVec3fTypes>::init();

// -- InteractionForceField interface
template <>
void SpringForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

//
// StiffSpringForceField
//

template <>
void StiffSpringForceField<gpu::cuda::CudaVec3fTypes>::init();

// -- InteractionForceField interface
template <>
void StiffSpringForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

template <>
void StiffSpringForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
