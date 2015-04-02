/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_GPU_CUDA_CUDABARYCENTRICMAPPINGRIGID_H
#define SOFA_GPU_CUDA_CUDABARYCENTRICMAPPINGRIGID_H

#include "CudaTypes.h"
#include <SofaMiscMapping/BarycentricMappingRigid.h>

namespace sofa
{

namespace component
{

namespace mapping
{


template<class TInReal, class TOutReal>
class BarycentricMapperTetrahedronSetTopology< gpu::cuda::CudaVectorTypes<sofa::defaulttype::Vec<3,TInReal>,sofa::defaulttype::Vec<3,TInReal>,TInReal>, sofa::defaulttype::StdRigidTypes<3,TOutReal> > : public BarycentricMapperTetrahedronSetTopologyRigid< gpu::cuda::CudaVectorTypes<sofa::defaulttype::Vec<3,TInReal>,sofa::defaulttype::Vec<3,TInReal>,TInReal>, sofa::defaulttype::StdRigidTypes<3,TOutReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<sofa::defaulttype::Vec<3,TInReal>,sofa::defaulttype::Vec<3,TInReal>,TInReal> In;
    typedef sofa::defaulttype::StdRigidTypes<3,TOutReal> Out;
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopology,In,Out),SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopologyRigid,In,Out));
    typedef BarycentricMapperTetrahedronSetTopologyRigid<In,Out> Inherit;

    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* _toTopology,
            helper::ParticleMask *_maskFrom,
            helper::ParticleMask *_maskTo)
        : Inherit(fromTopology, _toTopology, _maskFrom, _maskTo)
    {}

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
