/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_H

#include "CudaTypes.h"
#include <SofaDeformable/SpringForceField.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaDeformable/MeshSpringForceField.h>


namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class DataTypes>
class CudaKernelsSpringForceField;

} // namespace cuda

} // namespace gpu

namespace component
{

namespace interactionforcefield
{

template <class TCoord, class TDeriv, class TReal>
class SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef SpringForceField<DataTypes> Main;
    typedef typename Main::Inherit Inherit;
    typedef typename Main::Spring Spring;
    typedef SpringForceFieldInternalData<DataTypes> Data;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef gpu::cuda::CudaKernelsSpringForceField<DataTypes> Kernels;

    //enum { BSIZE=16 };
    struct GPUSpring
    {
        int index; ///< 0 if no spring
        //float initpos;
        float ks;
        //float kd;
        GPUSpring() : index(0), /*initpos(0),*/ ks(0)/*, kd(0)*/ {}
        void set(int index, float /*initpos*/, float ks, float /*kd*/)
        {
            this->index = index+1;
            //this->initpos = initpos;
            this->ks = ks;
            //this->kd = kd;
        }
    };
    struct GPUSpring2
    {
        //int index; ///< 0 if no spring
        float initpos;
        //float ks;
        float kd;
        GPUSpring2() : /*index(0),*/ initpos(0), /*ks(0),*/ kd(0) {}
        void set(int /*index*/, float initpos, float /*ks*/, float kd)
        {
            //this->index = index+1;
            this->initpos = initpos;
            //this->ks = ks;
            this->kd = kd;
        }
    };
    struct GPUSpringSet
    {
        int vertex0; ///< index of the first vertex connected to a spring
        int nbVertex; ///< number of vertices to process to compute all springs
        int nbSpringPerVertex; ///< max number of springs connected to a vertex
        gpu::cuda::CudaVector<GPUSpring> springs; ///< springs attached to each points (layout per bloc of NBLOC vertices, with first spring of each vertex, then second spring, etc)
        gpu::cuda::CudaVector<Real> dfdx; ///< only used for StiffSpringForceField
        GPUSpringSet() : vertex0(0), nbVertex(0), nbSpringPerVertex(0) {}
        void init(int v0, int nbv, int nbsperv)
        {
            vertex0 = v0;
            nbVertex = nbv;
            nbSpringPerVertex = nbsperv;
            int nbloc = (nbVertex+BSIZE-1)/BSIZE;
            springs.resize(2*nbloc*nbSpringPerVertex*BSIZE);
        }
        void set(int vertex, int spring, int index, float initpos, float ks, float kd)
        {
            int bloc = vertex/BSIZE;
            int b_x  = vertex%BSIZE;
            springs[ 2*bloc*BSIZE*nbSpringPerVertex // start of the bloc
                    + 2*spring*BSIZE                 // offset to the spring
                    + b_x                          // offset to the vertex
                   ].set(index, initpos, ks, kd);
            (*(GPUSpring2*)&(springs[ 2*bloc*BSIZE*nbSpringPerVertex // start of the bloc
                    + 2*spring*BSIZE                 // offset to the spring
                    + b_x+BSIZE                    // offset to the vertex
                                    ])).set(index, initpos, ks, kd);
        }
    };
    GPUSpringSet springs1; ///< springs from model1 to model2
    GPUSpringSet springs2; ///< springs from model2 to model1 (only used if model1 != model2)

    SpringForceFieldInternalData()
    {}

    static void init(Main* m, bool stiff);
    static void addForce(Main* m, bool stiff, VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);
    static void addDForce (Main* m, bool stiff, VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, SReal kFactor, SReal bFactor);
};

//
// SpringForceField
//

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaSpringForceField_DeclMethods(T) \
    template<> inline void SpringForceField< T >::init(); \
    template<> inline void SpringForceField< T >::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2); \
    template<> inline void StiffSpringForceField< T >::init(); \
    template<> inline void StiffSpringForceField< T >::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2); \
    template<> inline void StiffSpringForceField< T >::addDForce(const core::MechanicalParams*, DataVecDeriv& d_df1, DataVecDeriv& d_df2, const DataVecDeriv& d_dx1, const DataVecDeriv& d_dx2 );

CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaSpringForceField_DeclMethods

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
