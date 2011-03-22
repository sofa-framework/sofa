/*
 * CylinderMesh.inl
 *
 *  Created on: 21 mar. 2010
 *      Author: Yiyi
 */

#ifndef CGALPLUGIN_CYLINDERMESH_INL
#define CGALPLUGIN_CYLINDERMESH_INL
#include "CylinderMesh.h"

//#define CGAL_MESH_2_VERBOSE
//
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Constrained_Delaunay_triangulation_2.h>
//#include <CGAL/Constrained_triangulation_plus_2.h>
//#include <CGAL/Delaunay_mesher_2.h>
//#include <CGAL/Delaunay_mesh_face_base_2.h>
//#include <CGAL/Delaunay_mesh_size_criteria_2.h>
//#include <CGAL/Triangulation_vertex_base_with_id_2.h>
//#include <CGAL/Triangulation_vertex_base_with_info_2.h>
//#include <CGAL/Triangulation_face_base_with_info_2.h>
//// IO
//#include <CGAL/IO/Polyhedron_iostream.h>


//#define CINDEX(i, j, k, n) (i+j*n+k*n*n)
//#define PINDEX(i, j, k, n) (i+j*(n+1)+k*(n+1)*(n+1))
//
//#define PINDEX_0(i, j, k, n) PINDEX(i, j, k, n)
//#define PINDEX_1(i, j, k, n) PINDEX((i+1), j, k, n)
//#define PINDEX_2(i, j, k, n) PINDEX((i+1), (j+1), k, n)
//#define PINDEX_3(i, j, k, n) PINDEX(i, (j+1), k, n)
//#define PINDEX_4(i, j, k, n) PINDEX(i, j, (k+1), n)
//#define PINDEX_5(i, j, k, n) PINDEX((i+1), j, (k+1), n)
//#define PINDEX_6(i, j, k, n) PINDEX((i+1), (j+1), (k+1), n)
//#define PINDEX_7(i, j, k, n) PINDEX(i, (j+1), (k+1), n)

//CGAL
//struct K: public CGAL::Exact_predicates_inexact_constructions_kernel {};

using namespace sofa;

namespace cgal
{

template <class DataTypes>
CylinderMesh<DataTypes>::CylinderMesh()
    : m_diameter(initData(&m_diameter, 5.0, "diameter", "Diameter"))
    , m_length(initData(&m_length, 50.0, "length", "Length"))
    , m_number(initData(&m_number, 5, "number", "Number of intervals"))
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
    Real d = m_diameter.getValue();
    Real l = m_length.getValue();
    int n = m_number.getValue();
    if(d <=0 || l <=0 || n<=0)
    {
        std::cout << "ERROR: illegal parameters of the cylinder" << std::endl;
        return;
    }
    m_interval = d / n;
    int m = ceil(l/m_interval);
    l = m_interval * m;

    std::cout << "diameter = " << d << std::endl;
    std::cout << "length = " << l << std::endl;
    std::cout << "interval = " << m_interval << std::endl;

    n *= 2;
    m *= 2;
    Real t = m_interval / 2;

    std::cout << "n = " << n << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "t = " << t << std::endl;

    helper::WriteAccessor< Data< VecCoord > > points = m_points;
    helper::WriteAccessor< Data< SeqTetrahedra > > tetras = m_tetras;
    points.clear();
    tetras.clear();
    m_ptID.clear();

    //generate the points
    int count = 0;
    //hexa vertices
    for(int k = 0; k <= m; k+=2)
    {
        for(int j = 0; j <= n; j+=2)
        {
            for(int i = 0; i <= n; i+=2)
            {
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g,count));
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++count;
            }
        }
    }
    m_nbVertices = count;
    std::cout << "num of vertices = " << m_nbVertices << std::endl;

    //hexa centers
    for(int k = 1; k < m; k+=2)
    {
        for(int j = 1; j < n; j+=2)
        {
            for(int i = 1; i < n; i+=2)
            {
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g,count));
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++count;
            }
        }
    }
    m_nbCenters = count - m_nbVertices;
    std::cout << "num of centers = " << m_nbCenters << std::endl;

    //boundary centers
    //i = 0
    for(int k = 1; k <= m; k+=2)
    {
        for(int j = 1; j <= n; j+=2)
        {
            Point p(0, j*t, k*t);
            points.push_back(p);
            Index g(0,j,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //i = n
    for(int k = 1; k < m; k+=2)
    {
        for(int j = 1; j < n; j+=2)
        {
            Point p(d, j*t, k*t);
            points.push_back(p);
            Index g(n,j,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //j = 0
    for(int k = 1; k < m; k+=2)
    {
        for(int i = 1; i < n; i+=2)
        {
            Point p(i*t, 0, k*t);
            points.push_back(p);
            Index g(i,0,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //j = n
    for(int k = 1; k < m; k+=2)
    {
        for(int i = 1; i < n; i+=2)
        {
            Point p(i*t, d, k*t);
            points.push_back(p);
            Index g(i,n,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //k = 0
    for(int j = 1; j < n; j+=2)
    {
        for(int i = 1; i < n; i+=2)
        {
            Point p(i*t, j*t, 0);
            points.push_back(p);
            Index g(i,j,0);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //k = m
    for(int j = 1; j < n; j+=2)
    {
        for(int i = 1; i < n; i+=2)
        {
            Point p(i*t, j*t, l);
            points.push_back(p);
            Index g(i,j,m);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    m_nbBDCenters = count - m_nbVertices - m_nbCenters;
    std::cout << "num of boundary centers = " << m_nbBDCenters << std::endl;


    //generate tetrahedra between p(2i+1,2j+1,2k+1) and p(2i+3,2j+1,2k+1)
    for(int k = 0; k < m/2; ++k)
    {
        for(int j = 0; j < n/2; ++j)
        {
            for(int i = 0; i < n/2-1; ++i)
            {
                Index c1(2*i+1, 2*j+1, 2*k+1), c2(2*i+3, 2*j+1, 2*k+1);
                Index p1(2*i+2, 2*j, 2*k), p2(2*i+2, 2*j+2, 2*k), p3(2*i+2, 2*j+2, 2*k+2), p4(2*i+2, 2*j, 2*k+2);
                Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
                tetras.push_back(t1);
                Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
                tetras.push_back(t2);
                Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
                tetras.push_back(t3);
                Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
                tetras.push_back(t4);
            }
        }
    }
    //generate tetrahedra between p(2i+1,2j+1,2k+1) and p(2i+1,2j+3,2k+1)
    for(int k = 0; k < m/2; ++k)
    {
        for(int i = 0; i < n/2; ++i)
        {
            for(int j = 0; j < n/2-1; ++j)
            {
                Index c1(2*i+1, 2*j+1, 2*k+1), c2(2*i+1, 2*j+3, 2*k+1);
                Index p1(2*i, 2*j+2, 2*k), p2(2*i, 2*j+2, 2*k+2), p3(2*i+2, 2*j+2, 2*k+2), p4(2*i+2, 2*j+2, 2*k);
                Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
                tetras.push_back(t1);
                Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
                tetras.push_back(t2);
                Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
                tetras.push_back(t3);
                Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
                tetras.push_back(t4);
            }
        }
    }
    //generate tetrahedra between p(2i+1,2j+1,2k+1) and p(2i+1,2j+1,2k+3)
    for(int i = 0; i < n/2; ++i)
    {
        for(int j = 0; j < n/2; ++j)
        {
            for(int k = 0; k < m/2-1; ++k)
            {
                Index c1(2*i+1, 2*j+1, 2*k+1), c2(2*i+1, 2*j+1, 2*k+3);
                Index p1(2*i, 2*j, 2*k+2), p2(2*i, 2*j+2, 2*k+2), p3(2*i+2, 2*j+2, 2*k+2), p4(2*i+2, 2*j, 2*k+2);
                Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
                tetras.push_back(t1);
                Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
                tetras.push_back(t2);
                Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
                tetras.push_back(t3);
                Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
                tetras.push_back(t4);
            }
        }
    }
    //generate tetrahedra on the boundary i = 0
    for(int k = 0; k < m/2; ++k)
    {
        for(int j = 0; j < n/2; ++j)
        {
            Index c1(1, 2*j+1, 2*k+1), c2(0, 2*j+1, 2*k+1);
            Index p1(0, 2*j, 2*k), p2(0, 2*j+2, 2*k), p3(0, 2*j+2, 2*k+2), p4(0, 2*j, 2*k+2);
            Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
            tetras.push_back(t1);
            Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
            tetras.push_back(t2);
            Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
            tetras.push_back(t3);
            Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
            tetras.push_back(t4);
        }
    }
    //generate tetrahedra on the boundary i = n
    for(int k = 0; k < m/2; ++k)
    {
        for(int j = 0; j < n/2; ++j)
        {
            Index c1(n-1, 2*j+1, 2*k+1), c2(n, 2*j+1, 2*k+1);
            Index p1(n, 2*j, 2*k), p2(n, 2*j+2, 2*k), p3(n, 2*j+2, 2*k+2), p4(n, 2*j, 2*k+2);
            Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
            tetras.push_back(t1);
            Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
            tetras.push_back(t2);
            Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
            tetras.push_back(t3);
            Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
            tetras.push_back(t4);
        }
    }
    //generate tetrahedra on the boundary j = 0
    for(int k = 0; k < m/2; ++k)
    {
        for(int i = 0; i < n/2; ++i)
        {
            Index c1(2*i+1, 1, 2*k+1), c2(2*i+1, 0, 2*k+1);
            Index p1(2*i, 0, 2*k), p2(2*i, 0, 2*k+2), p3(2*i+2, 0, 2*k+2), p4(2*i+2, 0, 2*k);
            Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
            tetras.push_back(t1);
            Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
            tetras.push_back(t2);
            Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
            tetras.push_back(t3);
            Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
            tetras.push_back(t4);
        }
    }
    //generate tetrahedra on the boundary j = n
    for(int k = 0; k < m/2; ++k)
    {
        for(int i = 0; i < n/2; ++i)
        {
            Index c1(2*i+1, n-1, 2*k+1), c2(2*i+1, n, 2*k+1);
            Index p1(2*i, n, 2*k), p2(2*i, n, 2*k+2), p3(2*i+2, n, 2*k+2), p4(2*i+2, n, 2*k);
            Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
            tetras.push_back(t1);
            Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
            tetras.push_back(t2);
            Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
            tetras.push_back(t3);
            Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
            tetras.push_back(t4);
        }
    }
    //generate tetrahedra on the boundary k = 0
    for(int i = 0; i < n/2; ++i)
    {
        for(int j = 0; j < n/2; ++j)
        {
            Index c1(2*i+1, 2*j+1, 1), c2(2*i+1, 2*j+1, 0);
            Index p1(2*i, 2*j, 0), p2(2*i, 2*j+2, 0), p3(2*i+2, 2*j+2, 0), p4(2*i+2, 2*j, 0);
            Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
            tetras.push_back(t1);
            Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
            tetras.push_back(t2);
            Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
            tetras.push_back(t3);
            Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
            tetras.push_back(t4);
        }
    }
    //generate tetrahedra on the boundary k = m
    for(int i = 0; i < n/2; ++i)
    {
        for(int j = 0; j < n/2; ++j)
        {
            Index c1(2*i+1, 2*j+1, m-1), c2(2*i+1, 2*j+1, m);
            Index p1(2*i, 2*j, m), p2(2*i, 2*j+2, m), p3(2*i+2, 2*j+2, m), p4(2*i+2, 2*j, m);
            Tetra t1(m_ptID[c1], m_ptID[c2], m_ptID[p1], m_ptID[p2]);
            tetras.push_back(t1);
            Tetra t2(m_ptID[c1], m_ptID[c2], m_ptID[p2], m_ptID[p3]);
            tetras.push_back(t2);
            Tetra t3(m_ptID[c1], m_ptID[c2], m_ptID[p3], m_ptID[p4]);
            tetras.push_back(t3);
            Tetra t4(m_ptID[c1], m_ptID[c2], m_ptID[p4], m_ptID[p1]);
            tetras.push_back(t4);
        }
    }
    m_nbTetras = tetras.size();
    std::cout << "num of tetras = " << m_nbTetras << std::endl;



}

template <class DataTypes>
void CylinderMesh<DataTypes>::draw()
{
    if (m_viewPoints.getValue())
    {
        glDisable(GL_LIGHTING);
        helper::ReadAccessor< Data< VecCoord > > points = m_points;
        glPointSize(5);
        glBegin(GL_POINTS);
        //vertices
        glColor3f(0.0, 0.0, 1.0);
        for (int i = 0 ; i < m_nbVertices ; ++i)
            sofa::helper::gl::glVertexT(points[i]);
        //centers
        glColor3f(1.0, 0.0, 0.0);
        for (int i = m_nbVertices ; i < m_nbVertices+m_nbCenters ; ++i)
            sofa::helper::gl::glVertexT(points[i]);
        //boundary centers
        glColor3f(0.0, 1.0, 0.0);
        for (unsigned int i = m_nbVertices+m_nbCenters ; i < points.size(); ++i)
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
        glColor3f(1.0, 1.0, 1.0);
        glBegin(GL_LINES);
        for(int i = 0 ; i < m_nbTetras ; ++i)
        {
//        unsigned int i = 1;
            for(int j = 0 ; j < 3 ; ++j)
            {
                for(int k = j+1 ; k < 4 ; ++k)
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
