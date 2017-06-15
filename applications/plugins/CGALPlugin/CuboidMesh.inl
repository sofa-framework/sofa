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
 * CuboidMesh.inl
 *
 *  Created on: 12 sep. 2011
 *      Author: Yiyi
 */

#ifndef CGALPLUGIN_CUBOIDMESH_INL
#define CGALPLUGIN_CUBOIDMESH_INL
#include "CuboidMesh.h"


#define MAX(a,b) ( (a)>(b) ? (a):(b))
#define MIN(a,b) ( (a)<(b) ? (a):(b))

using namespace sofa;

namespace cgal
{

template <class DataTypes>
CuboidMesh<DataTypes>::CuboidMesh()
    : m_debug(initData(&m_debug, (unsigned)0, "debug", "for test"))
    , m_radius(initData(&m_radius, 5.0, "radius", "radius"))
    , m_height(initData(&m_height, 50.0, "height", "height"))
    , m_number(initData(&m_number, 5, "interval", "number of intervals"))
    , m_convex(initData(&m_convex, false, "convex", "make convex boundary"))
    , m_viewPoints(initData(&m_viewPoints, true, "viewPoints", "Display Points"))
    , m_viewTetras(initData(&m_viewTetras, true, "viewTetras", "Display Tetrahedra"))
    , m_points(initData(&m_points, "outputPoints", "Points"))
    , m_tetras(initData(&m_tetras, "outputTetras", "Tetrahedra"))
    , m_nbVertices(0), m_nbBdVertices(0), m_nbCenters(0), m_nbBdCenters(0)
    , m_nbTetras_i(0), m_nbTetras_j(0), m_nbTetras_k(0)
{
}

template <class DataTypes>
void CuboidMesh<DataTypes>::init()
{
    addOutput(&m_points);
    addOutput(&m_tetras);
    setDirtyValue();

    debug = m_debug.getValue();
    if(debug == 0)
        debug = ~debug;
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
    std::cout << "debug = " << debug << std::endl;
#endif
}

template <class DataTypes>
void CuboidMesh<DataTypes>::reinit()
{
    debug = m_debug.getValue();
    if(debug == 0)
        debug = ~debug;
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
    std::cout << "debug = " << debug << std::endl;
#endif
    update();
}

template <class DataTypes>
void CuboidMesh<DataTypes>::update()
{
    r = m_radius.getValue();
    h = m_height.getValue();
    n = m_number.getValue();
    if(r <=0 || h < 0 || n<=0)
    {
        std::cout << "ERROR: illegal parameters of the cuboid" << std::endl;
        return;
    }
    d = r / n;
    m = ceil(h / d);
    h = d * m;
    t = d / 2;

    std::cout << "radius = " << r << std::endl;
    std::cout << "height = " << h << std::endl;
    std::cout << "interval = " << d << std::endl;

    std::cout << "n = " << n << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "n_diameter = " << 2*n << std::endl;
    std::cout << "n_height = " << m+2*n << std::endl << std::endl;

    helper::WriteAccessor< Data< VecCoord > > points = m_points;
    helper::WriteAccessor< Data< SeqTetrahedra > > tetras = m_tetras;
    points.clear();
    tetras.clear();
    m_ptID.clear();

#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
    unsigned count = 0;
#endif

    //generate the points
    std::cout << "generate points..." << std::endl;
    //hexa vertices
    m_nbVertices = 0;
    for(int k = -m; k <= m; k+=2)
    {
        for(int j = -2*n; j <= 2*n; j+=2)
        {
            int begin = MAX(-2*n-j, -2*n+j);
            int end = MIN(2*n+j, 2*n-j);
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++m_nbVertices;
            }
        }
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
        for(int i = -2*n; i <= 2*n; i+=2)
        {
            int begin = MAX(-2*n-i, -2*n+i);
            int end = MIN(2*n+i, 2*n-i);
            for(int j = begin; j <= end; j+=2)
            {
                Index g(i,j,k);
                if(m_ptID.find(g) == m_ptID.end())
                {
                    std::cout << "ERROR: Vertices" << "(" << i << ", " << j << ", " << k << ")" << std::endl;
                }
                ++count;
            }
        }
#endif
    }
    std::cout << "num of vertices = " << m_nbVertices << " == (2n^2+2n+1)*(m+1) = " << (2*n*n+2*n+1)*(m+1) << std::endl;
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
    if(count != m_nbVertices || count != (unsigned)(2*n*n+2*n+1)*(m+1))
    {
        std::cout << "ERROR: num of vertices" << std::endl;
        std::cout << "count = " << count << std::endl;
    }
#endif

    //hexa centers
    m_nbCenters = 0;
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
    count = 0;
#endif
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int j = -2*n+1; j <= 2*n-1; j+=2)
        {
            int begin = MAX(-2*n-j, -2*n+j);
            int end = MIN(2*n+j, 2*n-j);
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                ++m_nbCenters;
            }
        }
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
        for(int i = -2*n+1; i <= 2*n-1; i+=2)
        {
            int begin = MAX(-2*n-i, -2*n+i);
            int end = MIN(2*n+i, 2*n-i);
            for(int j = begin; j <= end; j+=2)
            {
                Index g(i,j,k);
                if(m_ptID.find(g) == m_ptID.end())
                {
                    std::cout << "ERROR: Centers" << "(" << i << ", " << j << ", " << k << ")" << std::endl;
                }
                ++count;
            }
        }
#endif
    }
    std::cout << "num of centers = " << m_nbCenters << " == 2n(n+1)m = " << 2*n*(n+1)*m << std::endl;
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
    if(count != m_nbCenters || count != (unsigned)2*n*(n+1)*m)
    {
        std::cout << "ERROR: num of centers" << std::endl;
        std::cout << "count = " << count << std::endl;
    }
#endif

    if(m_convex.getValue())
    {
        //hexa boundary vertices
        m_nbBdVertices = 0;
        for(int k = m+2; k <= m+2*n; k+=2)
        {
            for(int j = -(2*n+m-k); j <= (2*n+m-k); j+=2)
            {
                int begin = MAX(-2*n-j, -2*n+j);
                int end = MIN(2*n+j, 2*n-j);
                for(int i = begin; i <= end; i+=2)
                {
                    Index g1(i,j,k);
                    m_ptID.insert(std::make_pair(g1, points.size()));
                    Point p1(i*t, j*t, k*t);
                    points.push_back(p1);
                    Index g2(i,j,-k);
                    m_ptID.insert(std::make_pair(g2, points.size()));
                    Point p2(i*t, j*t, -k*t);
                    points.push_back(p2);
                    m_nbBdVertices += 2;
                }
            }
        }
        std::cout << "num of bdVertices = " << m_nbBdVertices << std::endl;

#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
        double dif=0;
        count = 0;
        for(int k = -(m+2); k >= -(m+2*n); k-=2)
        {
            for(int j = -(2*n+m+k); j <= (2*n+m+k); j+=2)
            {
                int begin = MAX(-2*n-j, -2*n+j);
                int end = MIN(2*n+j, 2*n-j);
                for(int i = begin; i <= end; i+=2)
                {
                    Index g(i,j,k);
                    if(m_ptID.find(g) == m_ptID.end())
                    {
                        std::cout << "ERROR: bdVertices(k<=-(m+2))" << std::endl;
                    }
                    Point p(i*t, j*t, k*t);
                    dif += (points[m_ptID[g]] - p).norm();
                    ++count;
                }
            }
        }
        std::cout << "dif = " << dif << " count = " << count << " == bdVertices/2 = " << m_nbBdVertices/2.0 << std::endl;
#endif

        //boundary hexa centers
        m_nbBdCenters = 0;
#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
        count = 0;
#endif
        for(int k = m+1; k <= m+2*n-1; k+=2)
        {
            for(int j = -(2*n+m-k); j <= (2*n+m-k); j+=2)
            {
                int begin = MAX(-2*n-j, -2*n+j);
                int end = MIN(2*n+j, 2*n-j);
                for(int i = begin; i <= end; i+=2)
                {
                    Index g1(i,j,k);
                    m_ptID.insert(std::make_pair(g1, points.size()));
                    Point p1(i*t, j*t, k*t);
                    points.push_back(p1);
                    Index g2(i,j,-k);
                    m_ptID.insert(std::make_pair(g2, points.size()));
                    Point p2(i*t, j*t, -k*t);
                    points.push_back(p2);
                    m_nbBdCenters += 2;
                }
            }
        }
        std::cout << "num of bdCenters = " << m_nbBdCenters << std::endl;

#ifdef SOFA_CGALPLUGIN_CUBOIDMESH_DEBUG
        dif = 0, count = 0;
        for(int k = -(m+1); k >= -(m+2*n); k-=2)
        {
            for(int j = -(2*n+m+k); j <= (2*n+m+k); j+=2)
            {
                int begin = MAX(-2*n-j, -2*n+j);
                int end = MIN(2*n+j, 2*n-j);
                for(int i = begin; i <= end; i+=2)
                {
                    Index g(i,j,k);
                    if(m_ptID.find(g) == m_ptID.end())
                    {
                        std::cout << "ERROR: bdCenters(k<=-(m+1))" << std::endl;
                    }
                    Point p(i*t, j*t, k*t);
                    dif += (points[m_ptID[g]] - p).norm();
                    ++count;
                }
            }
        }
        std::cout << "dif = " << dif << " count = " << count << " == bdCenters/2 = " << m_nbBdCenters/2.0 << std::endl;
#endif
    }

    std::cout << "num of points = " << points.size() << std::endl << std::endl;

    std::cout << "generate tetras..." << std::endl;
    int span = m_convex.getValue() ? (m+2*n-1) : (m-1);
    //generate tetrahedra between c(i,j,k) and c(i+2,j,k) ((i+n), (j+n), (k+m) are odd numbers))
    m_nbTetras_i = 0;
    for(int k = -span; k <= span; k+=2)
    {
        for(int j = -2*n+1; j <= 2*n-1; j+=2)
        {
            for(int i = -2*n+1; i <= 2*n-1; i+=2)
            {
                Index c1(i,j,k), c2(i+2,j,k);
                if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                {
                    //std::cout << "tetrahedron is out of boundary. c "<< std::endl;
                    continue;
                }
                Index p[4] = {Index(i+1,j-1,k-1), Index(i+1,j+1,k-1), Index(i+1,j+1,k+1), Index(i+1,j-1,k+1)};
                for(int s = 0; s < 4; ++s)
                {
                    if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    {
                        //std::cout << "tetrahedron is out of boundary. p "<< std::endl;
                        continue;
                    }
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                    ++m_nbTetras_i;
                }
            }
        }
    }
    std::cout << "num of tetras_i = " << m_nbTetras_i << std::endl;

    //generate tetrahedra between c(i,j,k) and c(i,j+2,k) ((i+n), (j+n), (k+m) are odd numbers))
    m_nbTetras_j = 0;
    for(int k = -span; k <= span; k+=2)
    {
        for(int j = -2*n+1; j <= 2*n-1; j+=2)
        {
            for(int i = -2*n+1; i <= 2*n-1; i+=2)
            {
                Index c1(i,j,k), c2(i,j+2,k);
                if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                {
                    //std::cout << "tetrahedron is out of boundary. c"<< std::endl;
                    continue;
                }
                Index p[4] = {Index(i-1,j+1,k-1), Index(i+1,j+1,k-1), Index(i+1,j+1,k+1), Index(i-1,j+1,k+1)};
                for(int s = 0; s < 4; ++s)
                {
                    if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    {
                        //std::cout << "tetrahedron is out of boundary. p "<< std::endl;
                        continue;
                    }
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                    ++m_nbTetras_j;
                }
            }
        }
    }
    std::cout << "num of tetras_j = " << m_nbTetras_j << std::endl;

    //generate tetrahedra between c(i,j,k) and c(i,j,k+2) ((i+n), (j+n), (k+m) are odd numbers))
    m_nbTetras_k = 0;
    for(int k = -span; k <= span; k+=2)
    {
        for(int j = -2*n+1; j <= 2*n-1; j+=2)
        {
            for(int i = -2*n+1; i <= 2*n-1; i+=2)
            {
                Index c1(i,j,k), c2(i,j,k+2);
                Index p[4] = {Index(i-1,j-1,k+1), Index(i+1,j-1,k+1), Index(i+1,j+1,k+1), Index(i-1,j+1,k+1)};
                if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                {
                    //std::cout << "tetrahedron is out of boundary. c"<< std::endl;
                    continue;
                }
                for(int s = 0; s < 4; ++s)
                {
                    if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    {
                        //std::cout << "tetrahedron is out of boundary. p"<< std::endl;
                        continue;
                    }
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                    ++m_nbTetras_k;
                }
            }
        }
    }
    std::cout << "num of tetras_k = " << m_nbTetras_k << std::endl;
    std::cout << "num of tetras = " << tetras.size() << std::endl << std::endl;

    std::cout << "orientate..." << std::endl << std::endl;
    orientate();

    std::cout << "finished!" << std::endl << std::endl;
}

template <class DataTypes>
void CuboidMesh<DataTypes>::orientate()
{
    helper::ReadAccessor< Data< VecCoord > > points = m_points;
    helper::WriteAccessor< Data< SeqTetrahedra > > tetras = m_tetras;
    for(unsigned i = 0; i < tetras.size(); ++i)
    {
        Coord p[4];
        for(unsigned j = 0; j < 4; ++j)
        {
            p[j] = points[tetras[i][j]];
        }
        Coord p0p1 = p[1] - p[0];
        Coord p0p2 = p[2] - p[0];
        Coord p0p3 = p[3] - p[0];
        if(cross(p0p1, p0p2)*p0p3 < 0)
            std::swap(tetras[i][0], tetras[i][1]);
    }

}

template <class DataTypes>
void CuboidMesh<DataTypes>::draw()
{
    if (m_viewPoints.getValue())
    {
        glDisable(GL_LIGHTING);
        helper::ReadAccessor< Data< VecCoord > > points = m_points;
        glPointSize(8);
        glBegin(GL_POINTS);
        if(debug & 1)//1
        {
            //vertices
            //std::cout << "draw vertices" << std::endl;
            glColor3f(1.0, 0.0, 0.0);
            for(unsigned i = 0; i < m_nbVertices; ++i)
                sofa::helper::gl::glVertexT(points[i]);
        }
        if(debug & (1<<1))//2
        {
            //centers
            //std::cout << "draw centers" << std::endl;
            glColor3f(1.0, 1.0, 0.0);
            unsigned begin = m_nbVertices;
            unsigned end = m_nbVertices + m_nbCenters;
            for(unsigned i = begin; i < end; ++i)
                sofa::helper::gl::glVertexT(points[i]);
        }
        if(debug & (1<<2))//4
        {
            //centers
            //std::cout << "draw bdVertices" << std::endl;
            glColor3f(1.0, 0.0, 0.0);
            unsigned begin = m_nbVertices + m_nbCenters;
            unsigned end = m_nbVertices + m_nbCenters + m_nbBdVertices;
            for(unsigned i = begin; i < end; ++i)
                sofa::helper::gl::glVertexT(points[i]);
        }
        if(debug & (1<<3))//8
        {
            //bdCenters_i
            //std::cout << "draw bdCenters" << std::endl;
            glColor3f(1.0, 1.0, 0.0);
            unsigned begin = m_nbVertices + m_nbCenters + m_nbBdVertices;
            unsigned end = m_nbVertices + m_nbCenters + m_nbBdVertices + m_nbBdCenters;
            for(unsigned i = begin; i < end; ++i)
                sofa::helper::gl::glVertexT(points[i]);
        }
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
        if(debug & 1<<5)//32
        {
            for(unsigned i = 0; i < m_nbTetras_i; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    for(int k = j+1; k < 4; ++k)
                    {
                        sofa::helper::gl::glVertexT(points[tetras[i][j]]);
                        sofa::helper::gl::glVertexT(points[tetras[i][k]]);
                    }
                }
            }
        }
        if(debug & 1<<6)//64
        {
            for(unsigned i = m_nbTetras_i; i < m_nbTetras_i + m_nbTetras_j; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    for(int k = j+1; k < 4; ++k)
                    {
                        sofa::helper::gl::glVertexT(points[tetras[i][j]]);
                        sofa::helper::gl::glVertexT(points[tetras[i][k]]);
                    }
                }
            }
        }
        if(debug & 1<<7)//128
        {
            for(unsigned i = m_nbTetras_i + m_nbTetras_j; i < m_nbTetras_i + m_nbTetras_j + m_nbTetras_k; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    for(int k = j+1; k < 4; ++k)
                    {
                        sofa::helper::gl::glVertexT(points[tetras[i][j]]);
                        sofa::helper::gl::glVertexT(points[tetras[i][k]]);
                    }
                }
            }
        }
        glEnd();
        glEnable(GL_LIGHTING);
    }
}

} //cgal

#endif //CGALPLUGIN_CUBOIDMESH_INL
