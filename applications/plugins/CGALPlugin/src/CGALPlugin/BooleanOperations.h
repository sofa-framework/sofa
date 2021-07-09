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
#include <sofa/helper/OptionsGroup.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Surface_mesh.h>

namespace cgal
{

///
/// \brief The BooleanOperations class contains functions to corefine triangulated surface meshes and compute
/// triangulated surface meshes of the union, difference and intersection of the bounded volumes.
/// More info here: https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__corefinement__grp.html
///
class BooleanOperations : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(BooleanOperations,sofa::core::DataEngine);

    typedef typename sofa::defaulttype::Vec3Types::Real Real;
    typedef typename sofa::defaulttype::Vec3Types::Coord Point;
    typedef typename sofa::defaulttype::Vec3Types::Coord Coord;
    typedef typename sofa::defaulttype::Vec3Types::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef CGAL::Exact_predicates_exact_constructions_kernel ExactKernel;
    typedef Kernel::Point_3 Point_3;
    typedef Kernel::Vector_3 Vector_3;
    typedef CGAL::Surface_mesh<Kernel::Point_3> Mesh;
    typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
    typedef Mesh::Property_map<vertex_descriptor, ExactKernel::Point_3> Exact_point_map;

public:
    BooleanOperations();
    virtual ~BooleanOperations() { };

    void init() override;
    void doUpdate() override;

    //Inputs
    sofa::core::objectmodel::Data<sofa::helper::OptionsGroup> d_operation;
    sofa::core::objectmodel::Data<VecCoord> d_positions1In;
    sofa::core::objectmodel::Data<VecCoord> d_positions2In;
    sofa::core::objectmodel::Data<SeqTriangles> d_triangles1In;
    sofa::core::objectmodel::Data<SeqTriangles> d_triangles2In;
    sofa::core::objectmodel::Data<bool> d_computeDistribution;

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsOut; ///< Output positions of the surface mesh
    sofa::core::objectmodel::Data<SeqTriangles> d_trianglesOut; ///< Output triangles of the surface mesh
    sofa::core::objectmodel::Data<VecCoord> d_positions1Out; ///< Output positions of transformation on the first surface mesh
    sofa::core::objectmodel::Data<SeqTriangles> d_triangles1Out; ///< Output triangles of transformation on the first surface mesh
    sofa::core::objectmodel::Data<VecCoord> d_positions2Out; ///< Output positions of transformation on the second surface mesh
    sofa::core::objectmodel::Data<SeqTriangles> d_triangles2Out; ///< Output triangles of transformation the on second surface mesh
    sofa::core::objectmodel::Data<sofa::type::vector<int>> d_indices1Out; ///< Indices of the surface mesh points that are on the first object
    sofa::core::objectmodel::Data<sofa::type::vector<int>> d_indices2Out; ///< Indices of the surface mesh points that are on the second object

    sofa::type::vector<std::pair<int, std::pair<int, int>>> m_distribution;
    // List of pairs:
    //      - first: index in child
    //      - second: pair:
    //                    - first: parent index {0,1}
    //                    - second: index in parent

protected:
    bool checkMeshes(const CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3> &mesh1,
                     const CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3> &mesh2);
    void buildCGALMeshFromInput(const VecCoord &positions, const SeqTriangles &triangles,
                                CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3> &meshIn);
    void buildOutputFromCGALMesh(const CGAL::Surface_mesh<CGAL::Exact_predicates_inexact_constructions_kernel::Point_3>& meshOut,
                                 sofa::core::objectmodel::Data<VecCoord>& positions, sofa::core::objectmodel::Data<SeqTriangles>& triangles);
    void buildIndicesDistribution(const Exact_point_map& meshOut_exact_points,
                                  const Exact_point_map& mesh1_exact_points,
                                  const Exact_point_map& mesh2_exact_points);
};

} //cgal

