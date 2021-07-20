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
#include <CGALPlugin/FrontSurfaceReconstruction.h>

#include <CGAL/Advancing_front_surface_reconstruction.h>

#include <sofa/core/ObjectFactory.h>


int FrontSurfaceReconstructionClass = sofa::core::RegisterObject("Generate triangular surface mesh from point cloud")
        .add< cgal::FrontSurfaceReconstruction >()
        ;


namespace cgal
{

using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::Data;
using sofa::core::objectmodel::ComponentState ;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef std::array<std::size_t,3> Facet;

/// see: https://doc.cgal.org/latest/Advancing_front_surface_reconstruction/index.html#AFSR_Examples
struct Construct{
  Mesh& mesh;
  template < typename PointIterator>
  Construct(Mesh& mesh,PointIterator b, PointIterator e)
    : mesh(mesh)
  {
    for(; b!=e; ++b){
      boost::graph_traits<Mesh>::vertex_descriptor v;
      v = add_vertex(mesh);
      mesh.point(v) = *b;
    }
  }
  Construct& operator=(const Facet f)
  {
    typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
    typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
    mesh.add_face(vertex_descriptor(static_cast<size_type>(f[0])),
                  vertex_descriptor(static_cast<size_type>(f[1])),
                  vertex_descriptor(static_cast<size_type>(f[2])));
    return *this;
  }
  Construct&
  operator*() { return *this; }
  Construct&
  operator++() { return *this; }
  Construct
  operator++(int) { return *this; }
};


FrontSurfaceReconstruction::FrontSurfaceReconstruction()
    : d_positionsIn(initData (&d_positionsIn, "position", "Input point cloud positions"))
    , d_radiusRatioBound(initData (&d_radiusRatioBound, 5., "radiusRatioBound", "Candidates incident to surface triangles which are not in the beta-wedge are discarded, if the ratio of their radius and the radius of the surface triangle is larger than radius_ratio_bound"))
    , d_beta(initData (&d_beta, 0.52, "beta", "Half the angle of the wedge in which only the radius of triangles counts for the plausibility of candidates."))
    , d_positionsOut(initData (&d_positionsOut, "outputPosition", "Output position of the surface mesh"))
    , d_trianglesOut(initData (&d_trianglesOut, "outputTriangles", "Output triangles of the surface mesh"))
{
    addInput(&d_positionsIn);
    addInput(&d_radiusRatioBound);
    addInput(&d_beta);

    addOutput(&d_positionsOut);
    addOutput(&d_trianglesOut);

    setDirtyValue();
}


void FrontSurfaceReconstruction::init()
{
    d_componentState.setValue(ComponentState::Invalid);

    if(d_positionsIn.getValue().empty()){
        msg_error() << "No input positions. The component is disabled.";
        return;
    }

    d_componentState.setValue(ComponentState::Valid);
}


void FrontSurfaceReconstruction::doUpdate()
{
    if(d_componentState.getValue() == ComponentState::Invalid)
        return;

    ReadAccessor< Data<VecCoord> > positionsIn = d_positionsIn;

    std::vector<Point_3> points;

    for (sofa::Index i = 0; i<positionsIn.size() ; i++)
    {
        Point_3 p( positionsIn[i][0], positionsIn[i][1], positionsIn[i][2] );
        points.push_back(p);
    }

    Mesh meshOut;
    Construct construct(meshOut, points.begin(),points.end());

    CGAL::advancing_front_surface_reconstruction(points.begin(), points.end(), construct, d_radiusRatioBound.getValue(), d_beta.getValue());

    buildOutputFromCGALMesh(meshOut);
}


void FrontSurfaceReconstruction::buildOutputFromCGALMesh(Mesh& meshOut)
{
    WriteAccessor< Data<VecCoord> > positionOut = d_positionsOut;
    WriteAccessor< Data<SeqTriangles> > trianglesOut = d_trianglesOut;

    positionOut.clear();
    for (auto vId: meshOut.vertices())
    {
        Point_3 pointCgal = meshOut.point(vId);

        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());

        positionOut.push_back(p);
    }

    trianglesOut.clear();
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

        trianglesOut.push_back(t);
    }
}

} //cgal
 
