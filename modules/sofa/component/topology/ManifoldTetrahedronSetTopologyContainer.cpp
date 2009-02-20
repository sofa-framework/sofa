/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/component/topology/ManifoldTetrahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/MeshLoader.h>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ManidfoldTetrahedronSetTopologyContainer)
int ManifoldTetrahedronSetTopologyContainerClass = core::RegisterObject("Manifold Tetrahedron set topology container")
        .add< ManifoldTetrahedronSetTopologyContainer >()
        ;

const unsigned int tetrahedronEdgeArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

ManifoldTetrahedronSetTopologyContainer::ManifoldTetrahedronSetTopologyContainer()
    : TetrahedronSetTopologyContainer()// draw to be restored
    //, d_tetrahedron(initDataPtr(&d_tetrahedron, &m_tetrahedron, "tetras", "List of tetrahedron indices"))
    //, _draw(initData(&_draw, false, "drawTetras","if true, draw the tetrahedrons in the topology"))
{
}

ManifoldTetrahedronSetTopologyContainer::ManifoldTetrahedronSetTopologyContainer(const sofa::helper::vector< Tetrahedron >& tetrahedra )
    : TetrahedronSetTopologyContainer( tetrahedra)
    //, m_tetrahedron( tetrahedra )
    //, d_tetrahedron(initDataPtr(&d_tetrahedron, &m_tetrahedron, "tetras", "List of tetrahedron indices"))
{

}


void ManifoldTetrahedronSetTopologyContainer::init()
{
    TetrahedronSetTopologyContainer::init();
}


void ManifoldTetrahedronSetTopologyContainer::createTetrahedronVertexShellArray ()
{

    // TO be implemented
    // see late: for the topology, only one connexe composante around one vertex.

    TetrahedronSetTopologyContainer::createTetrahedronVertexShellArray();

}

void ManifoldTetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()
{

    // To be implemented :
    /*
      Tetrahedraons have to be oriented around each edges.
      Same algo as in 2d:

      - take the edge, third point find the next point in good order
      - use function getTetrahedronOrientation
      - loop
      - when map is done, order shell.
    */

    TetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray();

}

void ManifoldTetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()
{
    // To be implemented
    // at most 2 tetrahedrons adjacent to one triangle.


    TetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray();
}


bool ManifoldTetrahedronSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    // To be implemented later later....

    return ret && TetrahedronSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}

void ManifoldTetrahedronSetTopologyContainer::clear()
{
    //To be completed if necessary

    TetrahedronSetTopologyContainer::clear();
}


bool ManifoldTetrahedronSetTopologyContainer::getTetrahedronOrientation (const Tetrahedron &t, const Tetrahedron &t_test )
{
    //To be implemented
    /*

      First tetra is in one orientation. We know the 4 points
      we search the orientation of a second tetra.

      - First confirm it is the same 4 points
      - look how many permutation needed to fin the same tetra.
      - if nbr permuation is pair, same orientation

      => idea use 0 1 map to make bit a bit tests

     */
    //no warnings:
    (void) t;
    (void) t_test;

    return true;

}

int ManifoldTetrahedronSetTopologyContainer::getTriangleTetrahedronOrientation (const Tetrahedron &t, const Triangle &tri )
{
    //To be implemented

    /*

    - equivalent to TriangleEdgeShell [i]
    - first triangle of the tetrahedron should be in positive orientation
    - This first triangle is the one on the border if tetrahedron is on border.
    - return either negatif or positive orientation in the tetrahedron or -1 if error.

    => should be used in createTetrahedronTriangleShellArray



      for(TetraID i = 0; i < m_nbTetras; ++i)
    {
        const Tetra& t = m_topo->getTetra(i);
        const TetraTriangles& tFaces = m_topo->getTriangleTetraShell(i);
        for(int l = 0; l < 4; ++l)
        {
            int sign = 1;
            const Triangle& f = m_topo->getTriangle(tFaces[l]);

            int m = 0;
            while(t[m] == f[0] || t[m] == f[1] || t[m] == f[2])
                ++m;
            if(m%2 == 1)
                sign *= -1;

             int n = 0;
             while(f[0] != t[n])
                ++n;

            if((n+1)%4 == m && f[2] == t[(n+2)%4])
                sign *= -1;
            if((n+1)%4 != m && f[2] == t[(n+1)%4])
                sign *= -1;
        }
    }
    */

    //no warnings:
    (void) t;
    (void) tri;

    return 0;
}



} // namespace topology

} // namespace component

} // namespace sofa

