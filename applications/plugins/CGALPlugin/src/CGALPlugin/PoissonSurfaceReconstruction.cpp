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

#include <CGALPlugin/PoissonSurfaceReconstruction.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/property_map.h>

#include <sofa/core/ObjectFactory.h>

int PoissonSurfaceReconstructionClass = sofa::core::RegisterObject("Generate triangular surface mesh from point cloud")
        .add< cgal::PoissonSurfaceReconstruction >()
        ;

using namespace sofa;

namespace cgal
{

using sofa::core::objectmodel::ComponentState ;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef std::pair<Point_3, Vector_3> Pwn;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron_3;
typedef Polyhedron_3::Vertex_handle Vertex_handle;
typedef Polyhedron_3::Halfedge_around_facet_circulator HF_circulator;

PoissonSurfaceReconstruction::PoissonSurfaceReconstruction()
    : d_positionsIn(initData (&d_positionsIn, "position", "Input point cloud positions"))
    , d_normalsIn(initData (&d_normalsIn, "normals", "Input point cloud normals"))
    , d_angle(initData (&d_angle, 20., "angle", "Bound for the minimum facet angle in degrees"))
    , d_radius(initData (&d_radius, 30., "radius", "Bound for the radius of the surface Delaunay balls (relatively to the average_spacing)"))
    , d_distance(initData (&d_distance, 0.375, "distance", "Bound for the center-center distances (relatively to the average_spacing)"))
    , d_positionsOut(initData (&d_positionsOut, "outputPosition", "Output position of the surface mesh"))
    , d_trianglesOut(initData (&d_trianglesOut, "outputTriangles", "Output triangles of the surface mesh"))
{
    addInput(&d_positionsIn);
    addInput(&d_normalsIn);
    addInput(&d_angle);
    addInput(&d_radius);
    addInput(&d_distance);

    addOutput(&d_positionsOut);
    addOutput(&d_trianglesOut);

    setDirtyValue();
}


void PoissonSurfaceReconstruction::init()
{
    d_componentState.setValue(ComponentState::Invalid);

    if(d_positionsIn.getValue().empty()){
        msg_error() << "No input positions. The component is disabled.";
        return;
    }

    if(d_normalsIn.getValue().empty()){
        msg_error() << "No input normals. The component is disabled. Normals are required with PoissonSurfaceReconstruction, if you don't have normals consider using FrontSurfaceReconstruction instead.";
        return;
    }

    d_componentState.setValue(ComponentState::Valid);
}


void PoissonSurfaceReconstruction::doUpdate()
{
    if(d_componentState.getValue() == ComponentState::Invalid)
        return;

    helper::ReadAccessor< Data<VecCoord> > positionsIn = d_positionsIn;
    helper::ReadAccessor< Data<VecCoord> > normalsIn = d_normalsIn;
    helper::WriteAccessor< Data<VecCoord> > positionOut = d_positionsOut;
    helper::WriteAccessor< Data<SeqTriangles> > trianglesOut = d_trianglesOut;

    std::vector<Pwn> points;
    auto pointMap = CGAL::First_of_pair_property_map<Pwn>();
    auto normalMap = CGAL::Second_of_pair_property_map<Pwn>();

    for (sofa::Index i = 0; i<positionsIn.size() ; i++)
    {
        Point_3 p( positionsIn[i][0], positionsIn[i][1], positionsIn[i][2] );
        Vector_3 n( normalsIn[i][0], normalsIn[i][1], normalsIn[i][2] );
        points.push_back(std::pair(p,n));
        put(pointMap, *points.begin(), p);
        put(normalMap, *points.begin(), n);
    }

    double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()));

    Polyhedron_3 meshOut;
    CGAL::poisson_surface_reconstruction_delaunay(points.begin(), points.end(),
                                                  CGAL::First_of_pair_property_map<Pwn>(),
                                                  CGAL::Second_of_pair_property_map<Pwn>(), meshOut, average_spacing,
                                                  d_angle.getValue(), d_radius.getValue(), d_distance.getValue());

    int inum = 0;
    CGAL::Unique_hash_map<Vertex_handle, int > V;
    V.clear();

    positionOut.clear();
    for (auto vIt = meshOut.vertices_begin(); vIt != meshOut.vertices_end() ; ++vIt)
    {
        V[vIt] = inum++;
        Point_3 pointCgal = vIt->point();

        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());

        positionOut.push_back(p);
    }

    trianglesOut.clear();
    for (auto fIt = meshOut.facets_begin(); fIt != meshOut.facets_end() ; fIt++)
    {
        Triangle t;

        HF_circulator h = fIt->facet_begin();
        size_t order = 0;
        do
        {
            t[order] = V[h->vertex()];
            ++h;
            order++;
        }
        while( h != fIt->facet_begin()  && order < 3);

        trianglesOut.push_back(t);
    }
}

} //cgal
 
