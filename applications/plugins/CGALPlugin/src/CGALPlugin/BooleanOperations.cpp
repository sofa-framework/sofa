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

#include <CGALPlugin/BooleanOperations.h>

#include <sofa/core/ObjectFactory.h>

int BooleanOperationsClass = sofa::core::RegisterObject("Functions to corefine triangulated surface meshes and compute triangulated surface meshes of the union, difference and intersection of the bounded volumes.")
        .add< cgal::BooleanOperations >()
        ;

using namespace sofa;

namespace cgal
{

using sofa::core::objectmodel::ComponentState ;

/// See: https://doc.cgal.org/5.1.2/Polygon_mesh_processing/Polygon_mesh_processing_2corefinement_consecutive_bool_op_8cpp-example.html#a5
struct Exact_vertex_point_map
{
  // typedef for the property map
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef CGAL::Exact_predicates_exact_constructions_kernel ExactKernel;
    typedef Kernel::Point_3 Point_3;
    typedef CGAL::Surface_mesh<Kernel::Point_3> Mesh;
    typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
    typedef Mesh::Property_map<vertex_descriptor, ExactKernel::Point_3> Exact_point_map;
    typedef boost::property_traits<Exact_point_map>::value_type value_type;
    typedef boost::property_traits<Exact_point_map>::reference reference;
    typedef boost::property_traits<Exact_point_map>::category category;
    typedef boost::property_traits<Exact_point_map>::key_type key_type;
    // exterior references
    Exact_point_map exact_point_map;
    Mesh* tm_ptr;
    // Converters
    CGAL::Cartesian_converter<Kernel, ExactKernel> to_exact;
    CGAL::Cartesian_converter<ExactKernel, Kernel> to_input;
    Exact_vertex_point_map()
        : tm_ptr(nullptr)
    {}
    Exact_vertex_point_map(const Exact_point_map& ep, Mesh& tm)
        : exact_point_map(ep)
        , tm_ptr(&tm)
    {
        for (Mesh::Vertex_index v : vertices(tm))
            exact_point_map[v]=to_exact(tm.point(v));
    }
    friend
    reference get(const Exact_vertex_point_map& map, key_type k)
    {
        CGAL_precondition(map.tm_ptr!=nullptr);
        return map.exact_point_map[k];
    }
    friend
    void put(const Exact_vertex_point_map& map, key_type k, const ExactKernel::Point_3& p)
    {
        CGAL_precondition(map.tm_ptr!=nullptr);
        map.exact_point_map[k]=p;
        // create the input point from the exact one
        map.tm_ptr->point(k)=map.to_input(p);
    }
};


BooleanOperations::BooleanOperations()
    : d_operation(initData(&d_operation, sofa::helper::OptionsGroup(3,"union","intersection","difference"), "operation","Boolean operation"))
    , d_positions1In(initData (&d_positions1In, "position1", "Input positions of the first mesh"))
    , d_positions2In(initData (&d_positions2In, "position2", "Input positions of the second mesh"))
    , d_triangles1In(initData (&d_triangles1In, "triangles1", "Input triangles of the first mesh"))
    , d_triangles2In(initData (&d_triangles2In, "triangles2", "Input triangles of the second mesh"))
    , d_computeDistribution(initData (&d_computeDistribution, true, "computeDistrubution", "If true, computes outputIndices1 and outputIndices2"))
    , d_positionsOut(initData (&d_positionsOut, "outputPosition", "Output positions of the surface mesh"))
    , d_trianglesOut(initData (&d_trianglesOut, "outputTriangles", "Output triangles of the surface mesh"))
    , d_positions1Out(initData (&d_positions1Out, "outputPosition1", "Output positions of transformation on the first surface mesh"))
    , d_triangles1Out(initData (&d_triangles1Out, "outputTriangles1", "Output triangles of transformation on the first surface mesh"))
    , d_positions2Out(initData (&d_positions2Out, "outputPosition2", "Output positions of transformation on the second surface mesh"))
    , d_triangles2Out(initData (&d_triangles2Out, "outputTriangles2", "Output triangles of transformation on the second surface mesh"))
    , d_indices1Out(initData (&d_indices1Out, "outputIndices1", "Indices of the surface mesh points that are on the first object"))
    , d_indices2Out(initData (&d_indices2Out, "outputIndices2", "Indices of the surface mesh points that are on the second object"))
{
    addInput(&d_positions1In);
    addInput(&d_positions2In);
    addInput(&d_triangles1In);
    addInput(&d_triangles2In);
    addInput(&d_computeDistribution);

    addOutput(&d_positionsOut);
    addOutput(&d_trianglesOut);
    addOutput(&d_positions1Out);
    addOutput(&d_triangles1Out);
    addOutput(&d_positions2Out);
    addOutput(&d_triangles2Out);
    addOutput(&d_indices1Out);
    addOutput(&d_indices2Out);
    setDirtyValue();
}


void BooleanOperations::init()
{
    d_componentState.setValue(ComponentState::Invalid);

    if(d_positions1In.getValue().empty() || d_positions2In.getValue().empty()){
        msg_error() << "Missing input positions. The component is disabled.";
        return;
    }

    if(d_triangles1In.getValue().empty() || d_triangles2In.getValue().empty()){
        msg_error() << "Missing input triangles. The component is disabled.";
        return;
    }

    d_componentState.setValue(ComponentState::Valid);
}


void BooleanOperations::doUpdate()
{
    if(d_componentState.getValue() == ComponentState::Invalid)
        return;

    helper::ReadAccessor< Data<VecCoord> > positions1In = d_positions1In;
    helper::ReadAccessor< Data<VecCoord> > positions2In = d_positions2In;
    helper::ReadAccessor< Data<SeqTriangles> > triangles1In = d_triangles1In;
    helper::ReadAccessor< Data<SeqTriangles> > triangles2In = d_triangles2In;

    Mesh mesh1; buildCGALMeshFromInput(positions1In, triangles1In, mesh1);
    Mesh mesh2; buildCGALMeshFromInput(positions2In, triangles2In, mesh2);
    if(!checkMeshes(mesh1, mesh2)){
        d_componentState.setValue(ComponentState::Invalid);
        return;
    }

    Exact_point_map mesh1_exact_points = mesh1.add_property_map<vertex_descriptor,ExactKernel::Point_3>("e:exact_point").first;
    Exact_point_map mesh2_exact_points = mesh2.add_property_map<vertex_descriptor,ExactKernel::Point_3>("e:exact_point").first;

    Mesh meshOut;
    Exact_point_map meshOut_exact_points = meshOut.add_property_map<vertex_descriptor,ExactKernel::Point_3>("e:exact_point").first;

    Exact_vertex_point_map mesh1_vpm(mesh1_exact_points, mesh1);
    Exact_vertex_point_map mesh2_vpm(mesh2_exact_points, mesh2);
    Exact_vertex_point_map meshOut_vpm(meshOut_exact_points, meshOut);

    if(d_operation.getValue().getSelectedItem() == "union")
        CGAL::Polygon_mesh_processing::corefine_and_compute_union(mesh1, mesh2, meshOut,
                                                                  CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh1_vpm),
                                                                  CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh2_vpm),
                                                                  CGAL::Polygon_mesh_processing::parameters::vertex_point_map(meshOut_vpm));
    else if(d_operation.getValue().getSelectedItem() == "intersection")
        CGAL::Polygon_mesh_processing::corefine_and_compute_intersection(mesh1, mesh2, meshOut,
                                                                         CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh1_vpm),
                                                                         CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh2_vpm),
                                                                         CGAL::Polygon_mesh_processing::parameters::vertex_point_map(meshOut_vpm));
    else if(d_operation.getValue().getSelectedItem() == "difference")
        CGAL::Polygon_mesh_processing::corefine_and_compute_difference(mesh1, mesh2, meshOut,
                                                                       CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh1_vpm),
                                                                       CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh2_vpm),
                                                                       CGAL::Polygon_mesh_processing::parameters::vertex_point_map(meshOut_vpm));
    else
        return;

    buildOutputFromCGALMesh(meshOut, d_positionsOut, d_trianglesOut);
    buildOutputFromCGALMesh(mesh1, d_positions1Out, d_triangles1Out);
    buildOutputFromCGALMesh(mesh2, d_positions2Out, d_triangles2Out);
    if (d_computeDistribution.getValue())
        buildIndicesDistribution(meshOut_exact_points, mesh1_exact_points, mesh2_exact_points);
}


void BooleanOperations::buildCGALMeshFromInput(const VecCoord& positions,
                                               const SeqTriangles& triangles,
                                               Mesh& meshIn)
{
    meshIn.clear();

    for (auto position: positions){
        Point_3 p(position[0],position[1],position[2]);
        meshIn.add_vertex(p);
    }

    for (auto triangle: triangles){
        Mesh::Vertex_index id1 = Mesh::Vertex_index(triangle[0]);
        Mesh::Vertex_index id2 = Mesh::Vertex_index(triangle[1]);
        Mesh::Vertex_index id3 = Mesh::Vertex_index(triangle[2]);
        Mesh::Face_index f = meshIn.add_face(id1,id2,id3);
        if(f == Mesh::null_face())
            f = meshIn.add_face(id1,id3,id2);
    }
}


bool BooleanOperations::checkMeshes(const Mesh& mesh1, const Mesh& mesh2){
    bool status = true;
    if (CGAL::Polygon_mesh_processing::does_self_intersect(mesh1)){
        msg_error() << "First mesh self intersect"; status=false;
    }
    if (CGAL::Polygon_mesh_processing::does_self_intersect(mesh2)){
        msg_error() << "Second mesh self intersect"; status=false;
    }
    if (!CGAL::Polygon_mesh_processing::does_bound_a_volume(mesh1)){
        msg_error() << "First mesh does not bound a volume"; status=false;
    }
    if (!CGAL::Polygon_mesh_processing::does_bound_a_volume(mesh2)){
        msg_error() << "Second mesh does not bound a volume"; status=false;
    }
    return status;
}


void BooleanOperations::buildOutputFromCGALMesh(const Mesh& meshOut,
                                                Data<VecCoord>& positionsOut,
                                                Data<SeqTriangles>& trianglesOut)
{
    helper::WriteAccessor< Data<VecCoord> > positions = positionsOut;
    helper::WriteAccessor< Data<SeqTriangles> > triangles = trianglesOut;

    positions.clear();
    for (auto vId: meshOut.vertices())
    {
        Point_3 pointCgal = meshOut.point(vId);

        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());

        positions.push_back(p);
    }

    triangles.clear();
    for (auto tId: meshOut.faces())
    {
        Triangle t;

        CGAL::Vertex_around_face_circulator<Mesh> vcirc(meshOut.halfedge(tId), meshOut), done(vcirc);
        size_t order = 0;
        do
        {
            t[order] = *vcirc++;
            order++;
        }
        while( vcirc != done  && order < 3);

        triangles.push_back(t);
    }
}

void BooleanOperations::buildIndicesDistribution(const Exact_point_map &meshOut_exact_points,
                                                 const Exact_point_map &mesh1_exact_points,
                                                 const Exact_point_map &mesh2_exact_points)
{
    helper::WriteAccessor< Data<helper::vector<int>> > indices1 = d_indices1Out;
    helper::WriteAccessor< Data<helper::vector<int>> > indices2 = d_indices2Out;
    m_distribution.clear();
    std::pair<int, std::pair<int, int>> pair;

    indices1.clear();
    indices2.clear();
    for(auto itOut=meshOut_exact_points.begin(); itOut<meshOut_exact_points.end(); itOut++){
        int index = itOut - meshOut_exact_points.begin();
        for(auto it1=mesh1_exact_points.begin(); it1<mesh1_exact_points.end(); it1++){
            if(itOut->x()==it1->x() && itOut->y()==it1->y() && itOut->z()==it1->z()){
                pair.first = index;
                pair.second.first = 0;
                pair.second.second = it1 - mesh1_exact_points.begin();
                indices1.push_back(index);
                m_distribution.push_back(pair);
                break;
            }
        }
        for(auto it2=mesh2_exact_points.begin(); it2<mesh2_exact_points.end(); it2++){
            if(itOut->x()==it2->x() && itOut->y()==it2->y() && itOut->z()==it2->z()){
                pair.first = index;
                pair.second.first = 1;
                pair.second.second = it2 - mesh2_exact_points.begin();
                indices2.push_back(index);
                m_distribution.push_back(pair);
                break;
            }
        }
    }
}

} //cgal
 
