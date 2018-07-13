/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_VOLUMEGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_VOLUMEGEOMETRYALGORITHMS_H
#include "config.h"

#include <sofa/core/topology/CMBaseTopology.h>
#include <SofaBaseTopology/NumericalIntegrationDescriptor.h>

namespace sofa
{

namespace component
{

namespace topology
{


/**
* A class that provides geometry information on an TetrahedronSet.
*/
template < class DataTypes >
class VolumeGeometryAlgorithms : public core::cm_topology::GeometryAlgorithms
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VolumeGeometryAlgorithms,DataTypes),core::cm_topology::GeometryAlgorithms);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


protected:
    bool initializedCubatureTables;
    void defineTetrahedronCubaturePoints();

    VolumeGeometryAlgorithms()
        : Inherit1()
        ,initializedCubatureTables(false)
        , d_showVolumesIndices (initData(&d_showVolumesIndices, (bool) false, "showVolumesIndices", "Debug : view volumes indices"))
        , d_drawVolumes(initData(&d_drawVolumes, false, "drawVolumes","if true, draw the volumes in the topology"))
        , d_drawScaleVolumes(initData(&d_drawScaleVolumes, (float) 1.0, "drawScaleVolumes", "Scale of the terahedra (between 0 and 1; if <1.0, it produces gaps between the volumes)"))
        , d_drawColorVolumes(initData(&d_drawColorVolumes, sofa::defaulttype::Vec4f(1.0f,1.0f,0.0f,1.0f), "drawColorVolumes", "RGBA code color used to draw volumes."))
    {
    }

    virtual ~VolumeGeometryAlgorithms() {}
public:
    virtual void draw(const core::visual::VisualParams* vparams)
	{

	}

	bool isInTriangle(const Coord& P, const helper::fixed_array< Coord, 3 >& triangle)
	{
	  //    const Coord & AP = P - triangle[0];
	  //    const Coord & AB = triangle[1] - triangle[0];
	  //    const Coord & AC = triangle[2] - triangle[0];
	  //    const Coord & BA = -AB;
	  //    const SReal & alpha = AB * AP ;
	  //    const SReal & beta =  AC * AP ;
	  //    const SReal & gamma = BA * (P - triangle[1]);
	  //    return (alpha >= 0) && (beta >= 0) && (gamma >= 0);

	  const Coord& v2 = P - triangle[0];
	  const Coord& v1 = triangle[1] - triangle[0];
	  const Coord& v0 = triangle[2] - triangle[0];

	  const SReal& dot00 = v0 * v0;
	  const SReal& dot01 = v0 * v1;
	  const SReal& dot02 = v0 * v2;
	  const SReal& dot11 = v1 * v1;
	  const SReal& dot12 = v1 * v2;

	  const SReal& invDenom = static_cast< SReal >(1) / (dot00 * dot11 - dot01 * dot01);
	  const SReal& u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	  const SReal& v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	  return (u >= 0) && (v >= 0) && (u + v <= 1);
	}

protected:
    Data<bool> d_showVolumesIndices;
    Data<bool> d_drawVolumes;
    Data<float> d_drawScaleVolumes;
    Data<sofa::defaulttype::Vec4f> d_drawColorVolumes;
    /// include cubature points
//    NumericalIntegrationDescriptor<Real,4> tetrahedronNumericalIntegration;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API VolumeGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API VolumeGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API VolumeGeometryAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API VolumeGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API VolumeGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API VolumeGeometryAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_VOLUMEGEOMETRYALGORITHMS_H
