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
    , m_bScale(initData(&m_bScale, true, "scale", "Scale or not"))
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
    d = m_diameter.getValue();
    l = m_length.getValue();
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
    std::cout << "generate points..." << std::endl;
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

    std::cout << "generate tetras..." << std::endl;
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

    if(m_bScale.getValue())
    {
        std::cout << "scale..." << std::endl;
        scale();
    }

    std::cout << "orientate..." << std::endl;
    orientate();

    std::cout << "finished!" << std::endl;
}

template <class DataTypes>
void CylinderMesh<DataTypes>::scale()
{
    double lim = 2*(double)a/(double)n-1;
    helper::WriteAccessor< Data< VecCoord > > points = m_points;
    for (unsigned int i = 0; i < points.size(); ++i)
    {
        double x = points[i][0], y = points[i][1];
        if(fabs(y) < 1e-20)
            continue;
        double tg = x/y;
        if(tg>-1.0/lim && tg<-lim)
        {
            double k = d/(4*t*(double)a) * fabs(1-tg) / sqrt(1+tg*tg);
            //std::cout << "k = "  << k << std::endl;
            points[i][0] *= k, points[i][1] *= k;
            continue;
        }
        if(fabs(tg) <= lim)
        {
            double k = 1 / sqrt(1+tg*tg);
            //std::cout << "k = "  << k << std::endl;
            points[i][0] *= k, points[i][1] *= k;
            continue;
        }
        if(tg>lim && tg<1.0/lim)
        {
            double k = d/(4*t*(double)a) * fabs(1+tg) / sqrt(1+tg*tg);
            //std::cout << "k = "  << k << std::endl;
            points[i][0] *= k, points[i][1] *= k;
            continue;
        }
        if(fabs(tg)>=1.0/lim)
        {
            double k = 1 / sqrt(1+1.0/(tg*tg));
            //std::cout << "k = "  << k << std::endl;
            points[i][0] *= k, points[i][1] *= k;
            continue;
        }

    }
}

template <class DataTypes>
void CylinderMesh<DataTypes>::orientate()
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
void CylinderMesh<DataTypes>::draw()
{
    if (m_viewPoints.getValue())
    {
        glDisable(GL_LIGHTING);
        helper::ReadAccessor< Data< VecCoord > > points = m_points;
        glPointSize(8);
        glBegin(GL_POINTS);
        //vertices
        //glColor3f(1.0, 0.0, 0.0);
        glColor3f(0.0, 0.0, 1.0);
        for (int i = 0; i < m_nbVertices; ++i)
            sofa::helper::gl::glVertexT(points[i]);
        //centers
        //glColor3f(0.0, 0.0, 0.0);
        glColor3f(1.0, 0.0, 0.0);
        for (int i = m_nbVertices; i < m_nbVertices+m_nbCenters; ++i)
            sofa::helper::gl::glVertexT(points[i]);
        //boundary centers
        //glColor3f(0.0, 0.0, 0.0);
        glColor3f(0.0, 1.0, 0.0);
        for (unsigned int i = m_nbVertices+m_nbCenters; i < points.size(); ++i)
            sofa::helper::gl::glVertexT(points[i]);
        glEnd();

//        //bounding box
//        glColor3f(1.0, 0.0, 0.0);
//		glLineWidth(10);
//        glBegin(GL_LINE_LOOP);
//        glVertex3f((2*a-n)*t, n*t, -m*t);
//        glVertex3f(n*t, (2*a-n)*t, -m*t);
//        glVertex3f(n*t, (n-2*a)*t, -m*t);
//        glVertex3f((2*a-n)*t, -n*t, -m*t);
//        glVertex3f((n-2*a)*t, -n*t, -m*t);
//        glVertex3f(-n*t, (n-2*a)*t, -m*t);
//        glVertex3f(-n*t, (2*a-n)*t, -m*t);
//        glVertex3f((n-2*a)*t, n*t, -m*t);
//        glEnd();
//        glBegin(GL_LINE_LOOP);
//        glVertex3f((2*a-n)*t, n*t, m*t);
//        glVertex3f(n*t, (2*a-n)*t, m*t);
//        glVertex3f(n*t, (n-2*a)*t, m*t);
//        glVertex3f((2*a-n)*t, -n*t, m*t);
//        glVertex3f((n-2*a)*t, -n*t, m*t);
//        glVertex3f(-n*t, (n-2*a)*t, m*t);
//        glVertex3f(-n*t, (2*a-n)*t, m*t);
//        glVertex3f((n-2*a)*t, n*t, m*t);
//        glEnd();
//        glBegin(GL_LINES);
//        glVertex3f((2*a-n)*t, n*t, -m*t);
//        glVertex3f((2*a-n)*t, n*t, m*t);
//        glVertex3f(n*t, (2*a-n)*t, -m*t);
//        glVertex3f(n*t, (2*a-n)*t, m*t);
//        glVertex3f(n*t, (n-2*a)*t, -m*t);
//        glVertex3f(n*t, (n-2*a)*t, m*t);
//        glVertex3f((2*a-n)*t, -n*t, -m*t);
//        glVertex3f((2*a-n)*t, -n*t, m*t);
//        glVertex3f((n-2*a)*t, -n*t, -m*t);
//        glVertex3f((n-2*a)*t, -n*t, m*t);
//        glVertex3f(-n*t, (n-2*a)*t, -m*t);
//        glVertex3f(-n*t, (n-2*a)*t, m*t);
//        glVertex3f(-n*t, (2*a-n)*t, -m*t);
//        glVertex3f(-n*t, (2*a-n)*t, m*t);
//        glVertex3f((n-2*a)*t, n*t, -m*t);
//        glVertex3f((n-2*a)*t, n*t, m*t);
//        glEnd();
//
//		//circle
//		glColor3f(0.0, 0.0, 1.0);
//		int n = 1000;
//		float R = 0.5*m_diameter.getValue();
//		float L = 0.5*m_length.getValue();
//		float Pi = 3.1415926536f;
//		glBegin(GL_LINE_LOOP);
//		for(int i=0; i<n; ++i)
//         glVertex3f(R*cos(2*Pi/n*i), R*sin(2*Pi/n*i), L);
//		glEnd();
//		glLineWidth(1);

//        glBegin(GL_LINES);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f((2*a-n)*t, n*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f(n*t, (2*a-n)*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f(n*t, (n-2*a)*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f((2*a-n)*t, -n*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f((n-2*a)*t, -n*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f(-n*t, (n-2*a)*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f(-n*t, (2*a-n)*t, m*t);
//        glVertex3f(0.0, 0.0, m*t);
//        glVertex3f((n-2*a)*t, n*t, m*t);
//        glEnd();
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
