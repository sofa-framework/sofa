/*
 * CylinderMesh.inl
 *
 *  Created on: 21 mar. 2010
 *      Author: Yiyi
 */

#ifndef CGALPLUGIN_CYLINDERMESH_INL
#define CGALPLUGIN_CYLINDERMESH_INL
#include "CylinderMesh.h"

#define CGAL_MESH_2_VERBOSE

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_vertex_base_with_id_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
// IO
#include <CGAL/IO/Polyhedron_iostream.h>


#define CINDEX(i, j, k, n) (i+j*n+k*n*n)
#define PINDEX(i, j, k, n) (i+j*(n+1)+k*(n+1)*(n+1))

#define PINDEX_0(i, j, k, n) PINDEX(i, j, k, n)
#define PINDEX_1(i, j, k, n) PINDEX((i+1), j, k, n)
#define PINDEX_2(i, j, k, n) PINDEX((i+1), (j+1), k, n)
#define PINDEX_3(i, j, k, n) PINDEX(i, (j+1), k, n)
#define PINDEX_4(i, j, k, n) PINDEX(i, j, (k+1), n)
#define PINDEX_5(i, j, k, n) PINDEX((i+1), j, (k+1), n)
#define PINDEX_6(i, j, k, n) PINDEX((i+1), (j+1), (k+1), n)
#define PINDEX_7(i, j, k, n) PINDEX(i, (j+1), (k+1), n)

//CGAL
struct K: public CGAL::Exact_predicates_inexact_constructions_kernel {};

using namespace sofa;

namespace cgal
{
//    template <class DataTypes>
//    CylinderMesh<DataTypes>::CylinderMesh()
//    : f_points( initData (&f_points, "inputPoints", "Position coordinates (3D, z=0)"))
//                , f_edges(initData(&f_edges, "inputEdges", "Constraints (edges)"))
//                , f_edgesData1(initData(&f_edgesData1, "inputEdgesData1", "Data values defined on constrained edges"))
//                , f_edgesData2(initData(&f_edgesData2, "inputEdgesData2", "Data values defined on constrained edges"))
//                , f_seedPoints( initData (&f_seedPoints, "seedPoints", "Seed Points (3D, z=0)") )
//                , f_regionPoints( initData (&f_regionPoints, "regionPoints", "Region Points (3D, z=0)") )
//                , f_useInteriorPoints( initData (&f_useInteriorPoints, true, "useInteriorPoints", "should inputs points not on boundaries be input to the meshing algorithm"))
//                , f_newPoints( initData (&f_newPoints, "outputPoints", "New Positions coordinates (3D, z=0)") )
//                , f_newTriangles(initData(&f_newTriangles, "outputTriangles", "List of triangles"))
//                , f_newEdges(initData(&f_newEdges, "outputEdges", "New constraints (edges)"))
//                , f_newEdgesData1(initData(&f_newEdgesData1, "outputEdgesData1", "Data values defined on new constrained edges"))
//                , f_newEdgesData2(initData(&f_newEdgesData2, "outputEdgesData2", "Data values defined on new constrained edges"))
//                , f_trianglesRegion(initData(&f_trianglesRegion, "trianglesRegion", "Region for each Triangle"))
//                , f_newBdPoints(initData(&f_newBdPoints, "outputBdPoints", "Indices of points on the boundary"))
//                , p_shapeCriteria(initData(&p_shapeCriteria, 0.125, "shapeCriteria", "Shape Criteria"))
//                , p_sizeCriteria(initData(&p_sizeCriteria, 0.5, "sizeCriteria", "Size Criteria"))
//                , p_viewSeedPoints(initData(&p_viewSeedPoints, false, "viewSeedPoints", "Display Seed Points"))
//                , p_viewRegionPoints(initData(&p_viewRegionPoints, false, "viewRegionPoints", "Display Region Points"))*/
//    {
//
//    }

template <class DataTypes>
CylinderMesh<DataTypes>::CylinderMesh()
    : m_radius(initData(&m_radius, 5.0, "radius", "Radius"))
    , m_length(initData(&m_length, 50.0, "length", "Length"))
    , m_num(initData(&m_num, 5, "number", "Number of intervals"))
    , m_viewPoints(initData(&m_viewPoints, true, "viewPoints", "Display Points"))
    , m_viewTetras(initData(&m_viewTetras, true, "viewTetras", "Display Tetrahedra"))
    , m_points(initData(&m_points, "outputPoints", "Points"))
    , m_tetras(initData(&m_tetras, "outputTetras", "Tetrahedra"))
{
}

template <class DataTypes>
void CylinderMesh<DataTypes>::init()
{
    addOutput(&m_points);
    addOutput(&m_tetras);
    setDirtyValue();
}

template <class DataTypes>
void CylinderMesh<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void CylinderMesh<DataTypes>::update()
{
    Real r = m_radius.getValue();
    Real l = m_length.getValue();
    int n = m_num.getValue();
    if(r <=0 || l <=0 || n<=0)
    {
        std::cout << "ERROR: illegal parameters of the cylinder" << std::endl;
        return;
    }
    m_interval = r / (Real) n;
    int m = ceil(l/m_interval);
    l = m_interval * m;

    std::cout << "radius = " << r << std::endl;
    std::cout << "length = " << l << std::endl;
    std::cout << "interval = " << m_interval << std::endl;

    helper::WriteAccessor< Data< VecCoord > > points = m_points;
    helper::WriteAccessor< Data< SeqTetrahedra > > tetras = m_tetras;
    points.clear();
    tetras.clear();

    //generate the points
    VecCoord indices;
    //hexa vertices
    for(int k = 0; k <= m; ++k)
    {
        for(int j = 0; j <= n; ++j)
        {
            for(int i = 0; i <= n; ++i)
            {
                Point p(i*m_interval, j*m_interval, k*m_interval);
                Coord index(i, j, k);
                points.push_back(p);
                indices.push_back(index);
//                int count = points.size()-1;
//                if(count != i + j*n + k*n*n)
//                    std::cout << "ERROR: wrong index" << std::endl;
            }
        }
    }
    int numPoints = points.size();
    std::cout << "num of points = " << numPoints << std::endl;

    //hexa centers
    double offset = 0.5*m_interval;
    for(int k = 0; k < m; ++k)
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                Point p(i*m_interval+offset, j*m_interval+offset, k*m_interval+offset);
                Coord index(i, j, k);
                points.push_back(p);
                indices.push_back(index);
            }
        }
    }
    int numCenters = points.size() - numPoints;
    std::cout << "num of centers = " << numCenters << std::endl;

    //boundary centers
    //i = 0
    for(int k = 0; k < m; ++k)
    {
        for(int j = 0; j < n; ++j)
        {
            Point p(0.0, j*m_interval+offset, k*m_interval+offset);
            Coord index(0, j, k);
            points.push_back(p);
            indices.push_back(index);
        }
    }
    //i = n-1
    for(int k = 0; k < m; ++k)
    {
        for(int j = 0; j < n; ++j)
        {
            Point p(r, j*m_interval+offset, k*m_interval+offset);
            Coord index(n-1, j, k);
            points.push_back(p);
            indices.push_back(index);
        }
    }

    //generate tetrahedra between tetra(i,j,k) and tetra(i+1,j,k)
    for(int k = 0; k < m; ++k)
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n-1; ++i)
            {
                Tetra t1(CINDEX(i,j,k,n)+numPoints, CINDEX((i+1),j,k,n)+numPoints,
                        PINDEX_1(i,j,k,n), PINDEX_2(i,j,k,n));
                tetras.push_back(t1);
                Tetra t2(CINDEX(i,j,k,n)+numPoints, CINDEX((i+1),j,k,n)+numPoints,
                        PINDEX_2(i,j,k,n), PINDEX_6(i,j,k,n));
                tetras.push_back(t2);
                Tetra t3(CINDEX(i,j,k,n)+numPoints, CINDEX((i+1),j,k,n)+numPoints,
                        PINDEX_6(i,j,k,n), PINDEX_5(i,j,k,n));
                tetras.push_back(t3);
                Tetra t4(CINDEX(i,j,k,n)+numPoints, CINDEX((i+1),j,k,n)+numPoints,
                        PINDEX_5(i,j,k,n), PINDEX_1(i,j,k,n));
                tetras.push_back(t4);
            }
        }
    }

    //generate tetrahedra between tetra(i,j,k) and tetra(i,j+1,k)
    for(int k = 0; k < m; ++k)
    {
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n-1; ++j)
            {
                Tetra t1(CINDEX(i,j,k,n)+numPoints, CINDEX(i,(j+1),k,n)+numPoints,
                        PINDEX_2(i,j,k,n), PINDEX_3(i,j,k,n));
                tetras.push_back(t1);
                Tetra t2(CINDEX(i,j,k,n)+numPoints, CINDEX(i,(j+1),k,n)+numPoints,
                        PINDEX_3(i,j,k,n), PINDEX_7(i,j,k,n));
                tetras.push_back(t2);
                Tetra t3(CINDEX(i,j,k,n)+numPoints, CINDEX(i,(j+1),k,n)+numPoints,
                        PINDEX_7(i,j,k,n), PINDEX_6(i,j,k,n));
                tetras.push_back(t3);
                Tetra t4(CINDEX(i,j,k,n)+numPoints, CINDEX(i,(j+1),k,n)+numPoints,
                        PINDEX_6(i,j,k,n), PINDEX_2(i,j,k,n));
                tetras.push_back(t4);
            }
        }
    }

    //generate tetrahedra between tetra(i,j,k) and tetra(i,j,k+1)
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            for(int k = 0; k < m-1; ++k)
            {
                Tetra t1(CINDEX(i,j,k,n)+numPoints, CINDEX(i,j,(k+1),n)+numPoints,
                        PINDEX_6(i,j,k,n), PINDEX_7(i,j,k,n));
                tetras.push_back(t1);
                Tetra t2(CINDEX(i,j,k,n)+numPoints, CINDEX(i,j,(k+1),n)+numPoints,
                        PINDEX_7(i,j,k,n), PINDEX_4(i,j,k,n));
                tetras.push_back(t2);
                Tetra t3(CINDEX(i,j,k,n)+numPoints, CINDEX(i,j,(k+1),n)+numPoints,
                        PINDEX_4(i,j,k,n), PINDEX_5(i,j,k,n));
                tetras.push_back(t3);
                Tetra t4(CINDEX(i,j,k,n)+numPoints, CINDEX(i,j,(k+1),n)+numPoints,
                        PINDEX_5(i,j,k,n), PINDEX_6(i,j,k,n));
                tetras.push_back(t4);
            }
        }
    }

    int numTetras = tetras.size();
    std::cout << "num of tetras = " << numTetras << std::endl;

}

template <class DataTypes>
void CylinderMesh<DataTypes>::draw()
{
    int n = m_num.getValue();
    int m = m_length.getValue() / m_interval;
    if (m_viewPoints.getValue())
    {
        glDisable(GL_LIGHTING);

        const VecCoord& points = m_points.getValue();
        glPointSize(5);
        glColor3f(0.0, 0.0, 1.0);
        glBegin(GL_POINTS);
        for (int i = 0 ; i < (n+1)*(n+1)*(m+1) ; ++i)
            sofa::helper::gl::glVertexT(points[i]);
        glEnd();
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_POINTS);
        for (unsigned int i = (n+1)*(n+1)*(m+1) ; i < points.size() ; ++i)
            sofa::helper::gl::glVertexT(points[i]);
        glEnd();
        glPointSize(1);
        glEnable(GL_LIGHTING);
    }

    if (m_viewTetras.getValue())
    {
        helper::ReadAccessor< Data< VecCoord > > points = m_points;
        helper::ReadAccessor< Data< SeqTetrahedra > > tetras = m_tetras;

        glDisable(GL_LIGHTING);
        glColor3f(0.0, 1.0, 0.0);
        glBegin(GL_LINES);
        for(unsigned int i = 0 ; i < tetras.size() ; ++i)
        {
//        unsigned int i = 1;
            for(unsigned int j = 0 ; j < 3 ; ++j)
            {
                for(unsigned int k = j+1 ; k < 4 ; ++k)
                {
                    sofa::helper::gl::glVertexT(points[tetras[i][j]]);
                    sofa::helper::gl::glVertexT(points[tetras[i][k]]);
                }
            }
        }
        glEnd();
        glEnable(GL_LIGHTING);
    }


}

} //cgal

#endif //CGALPLUGIN_CYLINDERMESH_INL
