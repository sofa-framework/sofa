#ifndef SOFA_CONTRIB_CUDA_CUDASPRINGFORCEFIELD_H
#define SOFA_CONTRIB_CUDA_CUDASPRINGFORCEFIELD_H

#include "CudaTypes.h"
#include "Sofa-old/Components/SpringForceField.h"
#include "Sofa-old/Components/StiffSpringForceField.h"
#include "Sofa-old/Components/MeshSpringForceField.h"

namespace Sofa
{

namespace Components
{

template <>
class SpringForceFieldInternalData<Contrib::CUDA::CudaVec3fTypes>
{
public:
    enum { BSIZE=16 };
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
        Contrib::CUDA::CudaVector<GPUSpring> springs; ///< springs attached to each points (layout per bloc of NBLOC vertices, with first spring of each vertex, then second spring, etc)
        Contrib::CUDA::CudaVector<float> dfdx; ///< only used for StiffSpringForceField
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
void SpringForceField<Contrib::CUDA::CudaVec3fTypes>::init();

// -- InteractionForceField interface
template <>
void SpringForceField<Contrib::CUDA::CudaVec3fTypes>::addForce();

//
// StiffSpringForceField
//

template <>
void StiffSpringForceField<Contrib::CUDA::CudaVec3fTypes>::init();

// -- InteractionForceField interface
template <>
void StiffSpringForceField<Contrib::CUDA::CudaVec3fTypes>::addForce();

template <>
void StiffSpringForceField<Contrib::CUDA::CudaVec3fTypes>::addDForce();

} // namespace Components

} // namespace Sofa

#endif
