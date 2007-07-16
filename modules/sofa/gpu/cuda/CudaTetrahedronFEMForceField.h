#ifndef SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template <>
class TetrahedronFEMForceFieldInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    typedef gpu::cuda::CudaVec3fTypes::Real Real;

    /// Static data associated with each element
    struct GPUElement
    {
        /// index of the 4 connected vertices
        Vec<4,int> tetra;
        /// material stiffness matrix
        Mat<6,6,Real> K;
        /// strain-displacement matrix
        Mat<12,6,Real> J;
        /// initial position of the vertices in the local (rotated) coordinate system
        Vec3f initpos[4];
    };

    gpu::cuda::CudaVector<GPUElement> elems;


    /// Varying data associated with each element
    struct GPUElementState
    {
        /// rotation matrix
        Mat<3,3,Real> R;
        /// current internal strain
        Vec<6,Real> S;
        /// unused value to align to 64 bytes
        Real dummy;
    };

    gpu::cuda::CudaVector<GPUElementState> state;

    int vertex0; ///< index of the first vertex connected to an element
    int nbVertex; ///< number of vertices to process to compute all elements
    int nbElementPerVertex; ///< max number of elements connected to a vertex
    /// Index of elements attached to each points (layout per bloc of NBLOC vertices, with first element of each vertex, then second element, etc)
    /// No that each integer is actually equat the the index of the element * 4 + the index of this vertex inside the tetrahedron.
    gpu::cuda::CudaVector<int> velems;
    TetrahedronFEMForceFieldInternalData() : vertex0(0), nbVertex(0), nbElementPerVertex(0) {}
    void init(int v0, int nbv, int nbelemperv)
    {
        vertex0 = v0;
        nbVertex = nbv;
        nbElementPerVertex = nbelemperv;
        int nbloc = (nbVertex+BSIZE-1)/BSIZE;
        velems.resize(nbloc*nbElementPerVertex*BSIZE);
    }
    void set(int vertex, int num, int index)
    {
        int bloc = vertex/BSIZE;
        int b_x  = vertex%BSIZE;
        velems[ bloc*BSIZE*nbElementPerVertex // start of the bloc
                + num*BSIZE                     // offset to the element
                + b_x                           // offset to the vertex
              ] = index;
    }
};

//
// TetrahedronFEMForceField
//

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::reinit();

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx);

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
