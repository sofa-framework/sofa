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

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

namespace cgal
{

///
/// \brief The FrontSurfaceReconstruction class generates a surface mesh from a point cloud
/// More info here: https://doc.cgal.org/latest/Manual/tuto_reconstruction.html
///
class FrontSurfaceReconstruction : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(FrontSurfaceReconstruction,sofa::core::DataEngine);

    typedef typename sofa::defaulttype::Vec3Types::Real Real;
    typedef typename sofa::defaulttype::Vec3Types::Coord Point;
    typedef typename sofa::defaulttype::Vec3Types::Coord Coord;
    typedef typename sofa::defaulttype::Vec3Types::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;

public:
    FrontSurfaceReconstruction();
    virtual ~FrontSurfaceReconstruction() { };

    void init() override;
    void doUpdate() override;

    //Inputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsIn; ///< Input point cloud positions
    sofa::core::objectmodel::Data<double> d_radiusRatioBound; ///< Candidates incident to surface triangles which are not in the beta-wedge are discarded, if the ratio of their radius and the radius of the surface triangle is larger than radius_ratio_bound.
    sofa::core::objectmodel::Data<double> d_beta; ///< Half the angle of the wedge in which only the radius of triangles counts for the plausibility of candidates.

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsOut; ///< Output positions of the surface mesh
    sofa::core::objectmodel::Data<SeqTriangles> d_trianglesOut; ///< Output triangles of the surface mesh

protected:
    void buildOutputFromCGALMesh(CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3>& meshOut);
};

} //cgal

