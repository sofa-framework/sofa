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
#include <sofa/component/topology/ManifoldEdgeSetTopologyModifier.h>
#include <sofa/component/topology/EdgeSetTopologyChange.h>
#include <sofa/component/topology/ManifoldEdgeSetTopologyContainer.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(ManifoldEdgeSetTopologyModifier)
int ManifoldEdgeSetTopologyModifierClass = core::RegisterObject("ManifoldEdge set topology modifier")
        .add< ManifoldEdgeSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

void ManifoldEdgeSetTopologyModifier::init()
{
    EdgeSetTopologyModifier::init();
    getContext()->get(m_container);
}

void ManifoldEdgeSetTopologyModifier::addEdgeProcess(Edge e)
{
    m_container->resetConnectedComponent(); // invalidate the connected components by default

    EdgeSetTopologyModifier::addEdgeProcess(e);
}

void ManifoldEdgeSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    m_container->resetConnectedComponent(); // invalidate the connected components by default

    EdgeSetTopologyModifier::addEdgesProcess(edges);
}

void ManifoldEdgeSetTopologyModifier::removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    m_container->resetConnectedComponent(); // invalidate the connected components by default

    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}

} // namespace topology

} // namespace component

} // namespace sofa

