/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*
 * TriangularConvexHull3D.inl
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */
#ifndef CGALPLUGIN_TRIANGULARCONVEXHULL3D_INL
#define CGALPLUGIN_TRIANGULARCONVEXHULL3D_INL
#include "TriangularConvexHull3D.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/algorithm.h>
#include <CGAL/Convex_hull_traits_3.h>
#include <CGAL/convex_hull_3.h>


using namespace sofa;

namespace cgal
{

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef CGAL::Convex_hull_traits_3<K>             Traits;
typedef CGAL::Polyhedron_3<Traits> 			      Polyhedron_3;
typedef Polyhedron_3::Vertex 			   		  Vertex;
typedef Polyhedron_3::Vertex_iterator             Vertex_iterator;
typedef Polyhedron_3::Vertex_handle 		      Vertex_handle;
typedef Polyhedron_3::Facet_iterator              Facet_iterator;
typedef Polyhedron_3::Facet                       Facet;
typedef K::Segment_3                              Segment_3;

// define point creator
typedef K::Point_3                                Point_3;
typedef CGAL::Creator_uniform_3<double, Point_3>  PointCreator;
typedef Polyhedron_3::Halfedge_around_facet_circulator HF_circulator;

template <class DataTypes>
TriangularConvexHull3D<DataTypes>::TriangularConvexHull3D()
    : f_X0( initData (&f_X0, "inputPoints", "Rest position coordinates of the degrees of freedom") )
    , f_newX0( initData (&f_newX0, "outputPoints", "New Rest position coordinates") )
    , f_triangles(initData(&f_triangles, "outputTriangles", "List of triangles"))
{

}

template <class DataTypes>
void TriangularConvexHull3D<DataTypes>::init()
{
    addInput(&f_X0);

    addOutput(&f_newX0);
    addOutput(&f_triangles);

    setDirtyValue();
}

template <class DataTypes>
void TriangularConvexHull3D<DataTypes>::reinit()
{

}

template <class DataTypes>
void TriangularConvexHull3D<DataTypes>::update()
{
    helper::ReadAccessor< Data<VecCoord> > cloudPoints = f_X0;
    helper::WriteAccessor< Data<VecCoord> > newPoints = f_newX0;
    helper::WriteAccessor< Data<SeqTriangles> > newTriangles = f_triangles;

    typename VecCoord::const_iterator cloudPointsIt;

    if ((!newPoints.empty()) || cloudPoints.empty()) return;

    //CGAL::Random_points_in_sphere_3<Point_3, PointCreator> gen(100.0);
    std::vector<Point_3> points;
    for (cloudPointsIt = cloudPoints.begin() ; cloudPointsIt != cloudPoints.end() ; cloudPointsIt++)
    {
        //std::cout << (*cloudPointsIt) << std::endl;
        Point_3 p((*cloudPointsIt)[0], (*cloudPointsIt)[1],(*cloudPointsIt)[2] );
//		p[0] = (*cloudPointsIt)[0];
//		p[1] = (*cloudPointsIt)[1];
//		p[2] = (*cloudPointsIt)[2];

        points.push_back(p);
    }

    //CGAL::copy_n( gen, 2500, std::back_inserter(points) );

    // define object to hold convex hull
    CGAL::Object ch_object;

    // compute convex hull
    Polyhedron_3 P;
    CGAL::convex_hull_3(points.begin(), points.end(), P);

    newPoints.clear();

    Vertex_iterator vIt;
    Facet_iterator fIt;

    int inum = 0;
    CGAL::Unique_hash_map<Vertex_handle, int > V;
    V.clear();

    for (vIt = P.vertices_begin(); vIt != P.vertices_end() ; ++vIt)
    {
        V[vIt] = inum++;
        Point_3 pointCgal = vIt->point();

        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());

        newPoints.push_back(p);
    }

    newTriangles.clear();
    //triangles
    for (fIt = P.facets_begin(); fIt != P.facets_end() ; fIt++)
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

        newTriangles.push_back(t);
    }

}

} //cgal

#endif //CGALPLUGIN_TRIANGULARCONVEXHULL3D_INL
