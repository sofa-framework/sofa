/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H

#include "CudaTypes.h"
#include <sofa/component/container/MechanicalObject.h>

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

namespace container
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

    template<class T>
    class PrefetchOp : public T
    {
    public:
        int id; ///< ID in multi-operation, or -1 if inactive
        static helper::vector < Main* >& objects()
        {
            static helper::vector < Main* > v;
            return v;
        }
        PrefetchOp() : id(-1) {}
    };

    struct VDot
    {
        VecId a;
        VecId b;
        int size;
        double result;
    };
    PrefetchOp<VDot> preVDot;

    struct VOp
    {
        VecId v;
        VecId a;
        VecId b;
        double f;
        int size;
    };
    PrefetchOp< helper::vector<VOp> > preVOp;

    struct VResetForce
    {
        int size;
    };
    PrefetchOp< VResetForce > preVResetForce;

    static void accumulateForce(Main* m, bool prefetch = false);
    static void addDxToCollisionModel(Main* m, bool prefetch = false);
    static void vAlloc(Main* m, VecId v);
    static void vOp(Main* m, VecId v, VecId a, VecId b, double f, bool prefetch = false);
    static void vMultiOp(Main* m, const VMultiOp& ops, bool prefetch = false);
    static double vDot(Main* m, VecId a, VecId b, bool prefetch = false);
    static void resetForce(Main* m, bool prefetch = false);
};

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaMechanicalObject_DeclMethods(T) \
    template<> bool MechanicalObject< T >::canPrefetch() const; \
    template<> void MechanicalObject< T >::accumulateForce(); \
    template<> void MechanicalObject< T >::vOp(VecId v, VecId a, VecId b, double f); \
    template<> void MechanicalObject< T >::vMultiOp(const VMultiOp& ops); \
    template<> double MechanicalObject< T >::vDot(VecId a, VecId b); \
    template<> void MechanicalObject< T >::resetForce(); \
    template <> void MechanicalObject< T >::addDxToCollisionModel();

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaMechanicalObject_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaMechanicalObject_DeclMethods

}

} // namespace component

} // namespace sofa

#endif
