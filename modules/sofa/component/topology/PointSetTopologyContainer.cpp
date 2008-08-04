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
#include <sofa/component/topology/PointSetTopologyContainer.h>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/MeshLoader.h>

#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PointSetTopologyContainer)
int PointSetTopologyContainerClass = core::RegisterObject("Point set topology container")
        .add< PointSetTopologyContainer >()
        ;

PointSetTopologyContainer::PointSetTopologyContainer()
    : core::componentmodel::topology::TopologyContainer(),
      nbPoints(0)
{}

PointSetTopologyContainer::PointSetTopologyContainer(const int nPoints)
    : core::componentmodel::topology::TopologyContainer(),
      nbPoints(nPoints)
{}

void PointSetTopologyContainer::init()
{
    sofa::component::MeshLoader* loader;
    this->getContext()->get(loader);

    if(loader)
    {
        nbPoints = loader->getNbPoints();
    }
}

bool PointSetTopologyContainer::checkTopology() const
{
    return true;
}

void PointSetTopologyContainer::addPoints(const unsigned int nPoints)
{
    nbPoints += nPoints;
}

void PointSetTopologyContainer::removePoints(const unsigned int nPoints)
{
    nbPoints -= nPoints;
}

void PointSetTopologyContainer::addPoint(double , double , double )
{
    ++nbPoints;
}

void PointSetTopologyContainer::addPoint()
{
    ++nbPoints;
}

void PointSetTopologyContainer::removePoint()
{
    --nbPoints;
}

void PointSetTopologyContainer::clear()
{
    nbPoints = 0;
}

} // namespace topology

} // namespace component

} // namespace sofa

