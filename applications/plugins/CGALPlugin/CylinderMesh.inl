/*
 * CylinderMesh.inl
 *
 *  Created on: 21 mar. 2010
 *      Author: Yiyi
 */

#ifndef CGALPLUGIN_CYLINDERMESH_INL
#define CGALPLUGIN_CYLINDERMESH_INL
#include "CylinderMesh.h"


#define MAX(a,b) ( (a)>(b) ? (a):(b))
#define MIN(a,b) ( (a)<(b) ? (a):(b))

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
    n = m_number.getValue();
    if(d <=0 || l <=0 || n<=0)
    {
        std::cout << "ERROR: illegal parameters of the cylinder" << std::endl;
        return;
    }
    m_interval = d / n;
    m = ceil(l/m_interval);
    l = m_interval * m;
    t = m_interval / 2;
    a = ceil((d/2) / (sqrt(2)*t)); //parameters for cutting the corner

    std::cout << "diameter = " << d << std::endl;
    std::cout << "length = " << l << std::endl;
    std::cout << "interval = " << m_interval << std::endl;

    std::cout << "n = " << n << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "t = " << t << std::endl;
    std::cout << "a = " << a << std::endl;

    helper::WriteAccessor< Data< VecCoord > > points = m_points;
    helper::WriteAccessor< Data< SeqTetrahedra > > tetras = m_tetras;
    points.clear();
    tetras.clear();
    m_ptID.clear();

    //generate the points
    int count = 0;
    int b1, b2;
    //hexa vertices
    for(int k = -m; k <= m; k+=2)
    {
        for(int j = -n; j <= n; j+=2)
        {
            b1 = MAX(-n, MAX(-2*a-j, j-2*a)), b2 = MIN(n, MIN(2*a-j, j+2*a));
            for(int i = b1; i <= b2; i+=2)
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
    for(int k = -m+1; k < m; k+=2)
    {
        for(int j = -n+1; j < n; j+=2)
        {
            b1 = MAX(-n+1, MAX(-2*a-j, j-2*a)), b2 = MIN(n, MIN(2*a-j, j+2*a));
            for(int i = b1; i <= b2; i+=2)
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
    //i = -n
    b1 = MAX(-n+1, n-2*a+1), b2 = MIN(n, 2*a-n);
    for(int k = -m+1; k < m; k+=2)
    {
        for(int j = b1; j < b2; j+=2)
            //for(int j = -n+1; j < n; j+=2)
        {
            Point p(-n*t, j*t, k*t);
            points.push_back(p);
            Index g(-n,j,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //i = n
    for(int k = -m+1; k < m; k+=2)
    {
        for(int j = b1; j < b2; j+=2)
            //for(int j = -n+1; j < n; j+=2)
        {
            Point p(n*t, j*t, k*t);
            points.push_back(p);
            Index g(n,j,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //j = -n
    for(int k = -m+1; k < m; k+=2)
    {
        for(int i = b1; i < b2; i+=2)
            //for(int i = -n+1; i < n; i+=2)
        {
            Point p(i*t, -n*t, k*t);
            points.push_back(p);
            Index g(i,-n,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //j = n
    for(int k = -m+1; k < m; k+=2)
    {
        for(int i = b1; i < b2; i+=2)
            //for(int i = -n+1; i < n; i+=2)
        {
            Point p(i*t, n*t, k*t);
            points.push_back(p);
            Index g(i,n,k);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //k = -m
    for(int j = -n+1; j < n; j+=2)
    {
        b1 = MAX(-n+1, MAX(-2*a-j, -2*a+j)), b2 = MIN(n, MIN(2*a-j, 2*a+j));
        for(int i = b1; i <= b2; i+=2)
            //for(int i = -n+1; i < n; i+=2)
        {
            Point p(i*t, j*t, -m*t);
            points.push_back(p);
            Index g(i,j,-m);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    //k = m
    for(int j = -n+1; j < n; j+=2)
    {
        b1 = MAX(-n+1, MAX(-2*a-j, -2*a+j)), b2 = MIN(n, MIN(2*a-j, 2*a+j));
        for(int i = b1; i <= b2; i+=2)
            //for(int i = -n+1; i < n; i+=2)
        {
            Point p(i*t, j*t, m*t);
            points.push_back(p);
            Index g(i,j,m);
            m_ptID.insert(std::make_pair(g,count));
            //std::cout << "p[" << i/2 << ","<< j/2 << ","<< k/2 << "," << "] = " << p << std::endl;
            ++count;
        }
    }
    m_nbBDCenters = count - m_nbVertices - m_nbCenters;
    std::cout << "num of boundary centers = " << m_nbBDCenters << std::endl;

    //generate tetrahedra between c(i,j,k) and c(i+2,j,k) ((i+n), (j+n), (k+m) are odd numbers))
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int j = -n+1; j <= n-1; j+=2)
        {
            b1 = MAX(-n+1, MAX(-2*a-j, -2*a+j)), b2 = MIN(n-3, MIN(2*a-j-2, j+2*a-2));
            for(int i = b1; i <= b2; i+=2)
                //for(int i = -n+1; i < n-2; i+=2)
            {
                Index c1(i,j,k), c2(i+2,j,k);
                if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary. c "<< std::endl;
                Index p[4] = {Index(i+1,j-1,k-1), Index(i+1,j+1,k-1), Index(i+1,j+1,k+1), Index(i+1,j-1,k+1)};
                for(int s = 0; s < 4; ++s)
                {
                    if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                        std::cout << "ERROR: tetrahedron is out of boundary. p "<< std::endl;
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                }
            }
        }
    }
    //generate tetrahedra between c(i,j,k) and c(i,j+2,k) ((i+n), (j+n), (k+m) are odd numbers))
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int i = -n+1; i <= n-1; i+=2)
        {
            b1 = MAX(-n+1, MAX(-2*a-i, -2*a+i)), b2 = MIN(n-3, MIN(2*a-i-2, 2*a+i-2));
            for(int j = b1; j <= b2; j+=2)
                //for(int j = -n+1; j < n-2; j+=2)
            {
                Index c1(i,j,k), c2(i,j+2,k);
                if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary. c"<< std::endl;
                Index p[4] = {Index(i-1,j+1,k-1), Index(i+1,j+1,k-1), Index(i+1,j+1,k+1), Index(i-1,j+1,k+1)};
                for(int s = 0; s < 4; ++s)
                {
                    if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                        std::cout << "ERROR: tetrahedron is out of boundary. p"<< std::endl;
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                }
            }
        }
    }
    //generate tetrahedra between c(i,j,k) and c(i,j,k+2) ((i+n), (j+n), (k+m) are odd numbers))
    for(int i = -n+1; i <= n-1; i+=2)
    {
        b1 = MAX(-n+1, MAX(-2*a-i, -2*a+i)), b2 = MIN(n-1, MIN(2*a-i, 2*a+i));
        for(int j = b1; j <= b2; j+=2)
            //for(int j = -n+1; j < n; j+=2)
        {
            for(int k = -m+1; k <= m-3; k+=2)
            {
                Index c1(i,j,k), c2(i,j,k+2);
                Index p[4] = {Index(i-1,j-1,k+1), Index(i+1,j-1,k+1), Index(i+1,j+1,k+1), Index(i-1,j+1,k+1)};
                if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary. c"<< std::endl;
                bool flag[4] = {true, true, true, true};
                for(int s = 0; s < 4; ++s)
                    if(m_ptID.find(p[s]) == m_ptID.end())
                        flag[s] = false; //p[s] does not exist.
                for(int s = 0; s < 4; ++s)
                {
                    if(flag[s] && flag[(s+1)%4])
                    {
                        Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                        tetras.push_back(t);
                    }
                }
            }
        }
    }
    //generate tetrahedra on the boundary i = -n & i = n
    b1 = MAX(-n+1, n-2*a+1), b2 = MIN(n-1, 2*a-n-1);
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int j = b1; j <= b2; j+=2)
            //for(int j = -n+1; j <= n-1; j+=2)
        {
            Index c1(-n+1,j,k), c2(-n,j,k);
            if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
            Index p[4] = {Index(-n,j-1,k-1), Index(-n,j+1,k-1), Index(-n,j+1,k+1), Index(-n,j-1,k+1)};
            for(int s = 0; s < 4; ++s)
            {
                if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
                Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                tetras.push_back(t);
            }

        }
    }
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int j = b1; j <= b2; j+=2)
            //for(int j = -n+1; j < n; j+=2)
        {
            Index c1(n-1,j,k), c2(n,j,k);
            if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
            Index p[4] = {Index(n,j-1,k-1), Index(n,j+1,k-1), Index(n,j+1,k+1), Index(n,j-1,k+1)};
            for(int s = 0; s < 4; ++s)
            {
                if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
                Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                tetras.push_back(t);
            }

        }
    }
    //generate tetrahedra on the boundary j = -n & j = n
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int i = b1; i <= b2; i+=2)
            //for(int i = -n+1; i < n; i+=2)
        {
            Index c1(i,-n+1,k), c2(i,-n,k);
            if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
            Index p[4] = {Index(i-1,-n,k-1), Index(i+1,-n,k-1), Index(i+1,-n,k+1), Index(i-1,-n,k+1)};
            for(int s = 0; s < 4; ++s)
            {
                if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
                Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                tetras.push_back(t);
            }
        }
    }
    for(int k = -m+1; k <= m-1; k+=2)
    {
        for(int i = b1; i <= b2; i+=2)
            //for(int i = -n+1; i < n; i+=2)
        {
            Index c1(i,n-1,k), c2(i,n,k);
            if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
            Index p[4] = {Index(i-1,n,k-1), Index(i+1,n,k-1), Index(i+1,n,k+1), Index(i-1,n,k+1)};
            for(int s = 0; s < 4; ++s)
            {
                if(m_ptID.find(p[s]) == m_ptID.end() || m_ptID.find(p[(s+1)%4]) == m_ptID.end())
                    std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
                Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                tetras.push_back(t);
            }
        }
    }
    //generate tetrahedra on the boundary k = -m & k = m
    for(int i = -n+1; i <= n-1; i+=2)
    {
        b1 = MAX(-n+1, MAX(-2*a-i, -2*a+i)), b2 = MIN(n-1, MIN(2*a-i, 2*a+i));
        for(int j = b1; j <= b2; j+=2)
            //for(int j = -n+1; j < n; j+=2)
        {
            Index c1(i,j,-m+1), c2(i,j,-m);
            if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                std::cout << "ERROR: tetrahedron is out of boundary. c"<< std::endl;
            Index p[4] = {Index(i-1,j-1,-m), Index(i+1,j-1,-m), Index(i+1,j+1,-m), Index(i-1,j+1,-m)};
            bool flag[4] = {true, true, true, true};
            for(int s = 0; s < 4; ++s)
                if(m_ptID.find(p[s]) == m_ptID.end())
                {
                    flag[s] = false; //p[s] does not exist.
                    //std::cout << "false"<< std::endl;
                }
            for(int s = 0; s < 4; ++s)
            {
                if(flag[s] && flag[(s+1)%4])
                {
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                }
            }

        }
    }
    for(int i = -n+1; i <= n-1; i+=2)
    {
        b1 = MAX(-n+1, MAX(-2*a-i, -2*a+i)), b2 = MIN(n-1, MIN(2*a-i, 2*a+i));
        for(int j = b1; j <= b2; j+=2)
        {
            Index c1(i,j,m-1), c2(i,j,m);
            if(m_ptID.find(c1) == m_ptID.end() || m_ptID.find(c2) == m_ptID.end())
                std::cout << "ERROR: tetrahedron is out of boundary."<< std::endl;
            Index p[4] = {Index(i-1,j-1,m), Index(i+1,j-1,m), Index(i+1,j+1,m), Index(i-1,j+1,m)};
            bool flag[4] = {true, true, true, true};
            for(int s = 0; s < 4; ++s)
                if(m_ptID.find(p[s]) == m_ptID.end())
                {
                    flag[s] = false; //p[s] does not exist.
                    //std::cout << "false"<< std::endl;
                }
            for(int s = 0; s < 4; ++s)
            {
                if(flag[s] && flag[(s+1)%4])
                {
                    Tetra t(m_ptID[c1], m_ptID[c2], m_ptID[p[s]], m_ptID[p[(s+1)%4]]);
                    tetras.push_back(t);
                }
            }
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

        //bounding box
        glColor3f(1.0, 1.0, 1.0);
        glBegin(GL_LINE_LOOP);
        glVertex3f((2*a-n)*t, n*t, -m*t);
        glVertex3f(n*t, (2*a-n)*t, -m*t);
        glVertex3f(n*t, (n-2*a)*t, -m*t);
        glVertex3f((2*a-n)*t, -n*t, -m*t);
        glVertex3f((n-2*a)*t, -n*t, -m*t);
        glVertex3f(-n*t, (n-2*a)*t, -m*t);
        glVertex3f(-n*t, (2*a-n)*t, -m*t);
        glVertex3f((n-2*a)*t, n*t, -m*t);
        glEnd();
        glBegin(GL_LINE_LOOP);
        glVertex3f((2*a-n)*t, n*t, m*t);
        glVertex3f(n*t, (2*a-n)*t, m*t);
        glVertex3f(n*t, (n-2*a)*t, m*t);
        glVertex3f((2*a-n)*t, -n*t, m*t);
        glVertex3f((n-2*a)*t, -n*t, m*t);
        glVertex3f(-n*t, (n-2*a)*t, m*t);
        glVertex3f(-n*t, (2*a-n)*t, m*t);
        glVertex3f((n-2*a)*t, n*t, m*t);
        glEnd();
        glBegin(GL_LINES);
        glVertex3f((2*a-n)*t, n*t, -m*t);
        glVertex3f((2*a-n)*t, n*t, m*t);
        glVertex3f(n*t, (2*a-n)*t, -m*t);
        glVertex3f(n*t, (2*a-n)*t, m*t);
        glVertex3f(n*t, (n-2*a)*t, -m*t);
        glVertex3f(n*t, (n-2*a)*t, m*t);
        glVertex3f((2*a-n)*t, -n*t, -m*t);
        glVertex3f((2*a-n)*t, -n*t, m*t);
        glVertex3f((n-2*a)*t, -n*t, -m*t);
        glVertex3f((n-2*a)*t, -n*t, m*t);
        glVertex3f(-n*t, (n-2*a)*t, -m*t);
        glVertex3f(-n*t, (n-2*a)*t, m*t);
        glVertex3f(-n*t, (2*a-n)*t, -m*t);
        glVertex3f(-n*t, (2*a-n)*t, m*t);
        glVertex3f((n-2*a)*t, n*t, -m*t);
        glVertex3f((n-2*a)*t, n*t, m*t);
        glEnd();
        glPointSize(1);
        glEnable(GL_LIGHTING);
    }

    if (m_viewTetras.getValue())
    {
        helper::ReadAccessor< Data< VecCoord > > points = m_points;
        helper::ReadAccessor< Data< SeqTetrahedra > > tetras = m_tetras;

        glDisable(GL_LIGHTING);
        glColor3f(0.5, 0.5, 0.5);
        glBegin(GL_LINES);
        for(int i = 0; i < /*4*/m_nbTetras; ++i)
        {
//        unsigned int i = 1;
            for(int j = 0; j < 3; ++j)
            {
                for(int k = j+1; k < 4; ++k)
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
