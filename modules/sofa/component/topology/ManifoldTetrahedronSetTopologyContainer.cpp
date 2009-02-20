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

    TetrahedronSetTopologyContainer::createTetrahedronVertexShellArray();

}

void ManifoldTetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()
{

    // To be implemented

    TetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray();

}

void ManifoldTetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()
{
    // To be implemented

    TetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray();
}


bool ManifoldTetrahedronSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    // To be implemented

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



} // namespace topology

} // namespace component

} // namespace sofa

