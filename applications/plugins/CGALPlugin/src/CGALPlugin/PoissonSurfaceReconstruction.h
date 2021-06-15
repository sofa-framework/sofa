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
#pragma once

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace cgal
{

///
/// \brief The PoissonSurfaceReconstruction class generates a surface mesh from a point cloud
/// More info here: https://doc.cgal.org/latest/Manual/tuto_reconstruction.html
///
class PoissonSurfaceReconstruction : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(PoissonSurfaceReconstruction,sofa::core::DataEngine);

    typedef typename sofa::defaulttype::Vec3Types::Real Real;
    typedef typename sofa::defaulttype::Vec3Types::Coord Point;
    typedef typename sofa::defaulttype::Vec3Types::Coord Coord;
    typedef typename sofa::defaulttype::Vec3Types::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;

public:
    PoissonSurfaceReconstruction();
    virtual ~PoissonSurfaceReconstruction() { };

    void init() override;
    void doUpdate() override;

    //Inputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsIn; ///< Input point cloud positions
    sofa::core::objectmodel::Data<VecCoord> d_normalsIn; ///< Input point cloud normals
    sofa::core::objectmodel::Data<double> d_angle; ///< Bound for the minimum facet angle in degrees
    sofa::core::objectmodel::Data<double> d_radius; ///< Bound for the radius of the surface Delaunay balls (relatively to the average_spacing)
    sofa::core::objectmodel::Data<double> d_distance; ///< Bound for the center-center distances (relatively to the average_spacing)

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsOut; ///< Output positions of the surface mesh
    sofa::core::objectmodel::Data<SeqTriangles> d_trianglesOut; ///< Output triangles of the surface mesh
};

} //cgal

