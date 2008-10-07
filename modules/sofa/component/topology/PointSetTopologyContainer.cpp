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

PointSetTopologyContainer::PointSetTopologyContainer(int npoints)
    : nbPoints(npoints)
    , d_nbPoints(initDataPtr(&d_nbPoints, &nbPoints, "nbPoints", "Number of points"))
{
}

void PointSetTopologyContainer::setNbPoints(int n)
{
    nbPoints = n;
}

bool PointSetTopologyContainer::checkTopology() const
{
    return true;
}

void PointSetTopologyContainer::clear()
{
    nbPoints = 0;
    initPoints.clear();
}

void PointSetTopologyContainer::addPoint(double px, double py, double pz)
{
    initPoints.push_back(InitTypes::Coord((SReal)px, (SReal)py, (SReal)pz));
    if (initPoints.size() > (unsigned)nbPoints)
        nbPoints = initPoints.size();
}

bool PointSetTopologyContainer::hasPos() const
{
    return !initPoints.empty();
}

double PointSetTopologyContainer::getPX(int i) const
{
    if ((unsigned)i < initPoints.size())
        return initPoints[i][0];
    else
        return 0.0;
}

double PointSetTopologyContainer::getPY(int i) const
{
    if ((unsigned)i < initPoints.size())
        return initPoints[i][1];
    else
        return 0.0;
}

double PointSetTopologyContainer::getPZ(int i) const
{
    if ((unsigned)i < initPoints.size())
        return initPoints[i][2];
    else
        return 0.0;
}

void PointSetTopologyContainer::init()
{
    core::componentmodel::topology::TopologyContainer::init();

    if(nbPoints == 0)
    {
        sofa::component::MeshLoader* loader;
        this->getContext()->get(loader);

        if(loader)
        {
            loadFromMeshLoader(loader);
        }
    }
}

void PointSetTopologyContainer::loadFromMeshLoader(sofa::component::MeshLoader* loader)
{
    nbPoints = loader->getNbPoints();
}

void PointSetTopologyContainer::addPoints(const unsigned int nPoints)
{
    nbPoints += nPoints;
}

void PointSetTopologyContainer::removePoints(const unsigned int nPoints)
{
    nbPoints -= nPoints;
}

void PointSetTopologyContainer::addPoint()
{
    ++nbPoints;
}

void PointSetTopologyContainer::removePoint()
{
    --nbPoints;
}

} // namespace topology

} // namespace component

} // namespace sofa

