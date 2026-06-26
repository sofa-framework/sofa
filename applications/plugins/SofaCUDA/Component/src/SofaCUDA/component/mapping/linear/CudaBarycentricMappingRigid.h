/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_GPU_CUDA_CUDABARYCENTRICMAPPINGRIGID_H
#define SOFA_GPU_CUDA_CUDABARYCENTRICMAPPINGRIGID_H

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/mapping/linear/BarycentricMappingRigid.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace mapping::linear
{


template<class TInReal, class TOutReal>
class BarycentricMapperTetrahedronSetTopology< gpu::cuda::CudaVectorTypes<sofa::type::Vec<3,TInReal>,sofa::type::Vec<3,TInReal>,TInReal>, sofa::defaulttype::StdRigidTypes<3,TOutReal> > : public linear::BarycentricMapperTetrahedronSetTopologyRigid< gpu::cuda::CudaVectorTypes<sofa::type::Vec<3,TInReal>,sofa::type::Vec<3,TInReal>,TInReal>, sofa::defaulttype::StdRigidTypes<3,TOutReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<sofa::type::Vec<3,TInReal>,sofa::type::Vec<3,TInReal>,TInReal> In;
    typedef sofa::defaulttype::StdRigidTypes<3,TOutReal> Out;
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopology,In,Out),SOFA_TEMPLATE2(linear::BarycentricMapperTetrahedronSetTopologyRigid,In,Out));
    typedef linear::BarycentricMapperTetrahedronSetTopologyRigid<In,Out> Inherit;

    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* _toTopology)
        : Inherit(fromTopology, _toTopology)
    {}

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
