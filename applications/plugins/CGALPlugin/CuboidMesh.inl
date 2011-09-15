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
    :  m_debug(initData(&m_debug, (unsigned)0, "debug", "for test"))
    , m_radius(initData(&m_radius, 5.0, "radius", "radius"))
    , m_height(initData(&m_height, 50.0, "height", "height"))
    , m_number(initData(&m_number, 5, "interval", "number of intervals"))
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
    std::cout << "debug = " << debug << std::endl;
}

template <class DataTypes>
void CuboidMesh<DataTypes>::reinit()
{
    debug = m_debug.getValue();
    if(debug == 0)
        debug = ~debug;
    std::cout << "debug = " << debug << std::endl;

    //update();
}

template <class DataTypes>
void CuboidMesh<DataTypes>::update()
{
    r = m_radius.getValue();
    h = m_height.getValue();
    n = m_number.getValue();
    if(r <=0 || h <=0 || n<=0)
    {
        std::cout << "ERROR: illegal parameters of the cuboid" << std::endl;
        return;
    }
    d = r / n;
    m = ceil(h/d);
    h = d * m;
    t = d / 2;

    std::cout << "radius = " << r << std::endl;
    std::cout << "height = " << h << std::endl;
    std::cout << "interval = " << d << std::endl;

    std::cout << "n = " << 2*n << std::endl;
    std::cout << "m = " << m << std::endl << std::endl;

    helper::WriteAccessor< Data< VecCoord > > points = m_points;
    helper::WriteAccessor< Data< SeqTetrahedra > > tetras = m_tetras;
    points.clear();
    tetras.clear();
    m_ptID.clear();

    //generate the points
    std::cout << "generate points..." << std::endl;
    //hexa vertices
    m_nbVertices = 0;
    for(int k = -m; k <= m; k+=2)
    {
        for(int j = -2*n; j <= 2*n; j+=2)
        {
            int begin = MAX(-2*n, MAX(-2*n-j, -2*n+j));
            int end = MIN(2*n, MIN(2*n+j, 2*n-j));
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
    }
    std::cout << "num of vertices = " << m_nbVertices << std::endl;

    //hexa boundary vertices
    m_nbBdVertices = 0;
    for(int k = m+2; k <= m+2*n; k+=2)
    {
        for(int j = -(2*n+m-k); j <= (2*n+m-k); j+=2)
        {
            int begin = MAX(-2*n, MAX(-2*n-j, -2*n+j));
            int end = MIN(2*n, MIN(2*n+j, 2*n-j));
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++m_nbBdVertices;
            }
        }
    }
    for(int k = -(m+2); k >= -(m+2*n); k-=2)
    {
        for(int j = -(2*n+m+k); j <= (2*n+m+k); j+=2)
        {
            int begin = MAX(-2*n, MAX(-2*n-j, -2*n+j));
            int end = MIN(2*n, MIN(2*n+j, 2*n-j));
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++m_nbBdVertices;
            }
        }
    }
    std::cout << "num of bdVertices = " << m_nbBdVertices << std::endl;

    //hexa centers
    m_nbCenters = 0;
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int j = -2*n+1; j <= 2*n-1; j+=2)
        {
            int begin = MAX(-2*n+1, MAX(-2*n-j, -2*n+j));
            int end = MIN(2*n-1, MIN(2*n+j, 2*n-j));
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++m_nbCenters;
            }
        }
    }
    std::cout << "num of centers = " << m_nbCenters << std::endl;

    m_nbBdCenters = 0;
    for(int k = m+1; k <= m+2*n; k+=2)
    {
        for(int j = -(2*n+m-k); j <= (2*n+m-k); j+=2)
        {
            int begin = MAX(-2*n+1, MAX(-2*n-j, -2*n+j));
            int end = MIN(2*n-1, MIN(2*n+j, 2*n-j));
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++m_nbBdCenters;
            }
        }
    }
    for(int k = -(m+1); k >= -(m+2*n); k-=2)
    {
        for(int j = -(2*n+m+k); j <= (2*n+m+k); j+=2)
        {
            int begin = MAX(-2*n+1, MAX(-2*n-j, -2*n+j));
            int end = MIN(2*n-1, MIN(2*n+j, 2*n-j));
            for(int i = begin; i <= end; i+=2)
            {
                Index g(i,j,k);
                m_ptID.insert(std::make_pair(g, points.size()));
                Point p(i*t, j*t, k*t);
                points.push_back(p);
                //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
                ++m_nbBdCenters;
            }
        }
    }
    std::cout << "num of bdCenters = " << m_nbBdCenters << std::endl;
    std::cout << "num of points = " << points.size() << std::endl << std::endl;

    std::cout << "generate tetras..." << std::endl;
    //generate tetrahedra between c(i,j,k) and c(i+2,j,k) ((i+n), (j+n), (k+m) are odd numbers))
    m_nbTetras_i = 0;
    for(int k = -(m+2*n-1); k <= (m+2*n-1); k+=2)
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
    for(int k = -(m+2*n-1); k <= (m+2*n-1); k+=2)
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
    for(int k = -(m+2*n-1); k <= (m+2*n-1); k+=2)
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
    std::cout << "num of tetras = " << tetras.size() << std::endl;

    std::cout << "orientate..." << std::endl;
    orientate();

    std::cout << "finished!" << std::endl;
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
            //std::cout << "draw bdVertices" << std::endl;
            glColor3f(1.0, 0.0, 0.0);
            unsigned begin = m_nbVertices;
            unsigned end = m_nbVertices + m_nbBdVertices;
            for(unsigned i = begin; i < end; ++i)
                sofa::helper::gl::glVertexT(points[i]);
        }
        if(debug & (1<<2))//4
        {
            //centers
            //std::cout << "draw centers" << std::endl;
            glColor3f(1.0, 1.0, 0.0);
            unsigned begin = m_nbVertices + m_nbBdVertices;
            unsigned end = m_nbVertices + m_nbBdVertices + m_nbCenters;
            for(unsigned i = begin; i < end; ++i)
                sofa::helper::gl::glVertexT(points[i]);
        }
        if(debug & (1<<3))//8
        {
            //bdCenters_i
            //std::cout << "draw bdCenters" << std::endl;
            glColor3f(1.0, 1.0, 0.0);
            unsigned begin = m_nbVertices + m_nbBdVertices + m_nbCenters;
            unsigned end = m_nbVertices + m_nbBdVertices + m_nbCenters + m_nbBdCenters;
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
        glColor3f(0, 0, 0);
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
