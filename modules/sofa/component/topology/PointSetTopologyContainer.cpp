/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/simulation/common/Node.h>
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
    : nbPoints (initData(&nbPoints, (unsigned int )npoints, "nbPoints", "Number of points"))
    , d_initPoints (initData(&d_initPoints, "position", "Initial position of points"))
{
    addAlias(&d_initPoints,"points");
}

void PointSetTopologyContainer::setNbPoints(int n)
{
    nbPoints.setValue(n);
}

unsigned int PointSetTopologyContainer::getNumberOfElements() const
{
    return nbPoints.getValue();
}

bool PointSetTopologyContainer::checkTopology() const
{
    return true;
}

void PointSetTopologyContainer::clear()
{
    nbPoints.setValue(0);
    helper::WriteAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    initPoints.clear();
}

void PointSetTopologyContainer::addPoint(double px, double py, double pz)
{
    helper::WriteAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    initPoints.push_back(InitTypes::Coord((SReal)px, (SReal)py, (SReal)pz));
    if (initPoints.size() > nbPoints.getValue())
        nbPoints.setValue(initPoints.size());
}

bool PointSetTopologyContainer::hasPos() const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    return !initPoints.empty();
}

double PointSetTopologyContainer::getPX(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][0];
    else
        return 0.0;
}

double PointSetTopologyContainer::getPY(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][1];
    else
        return 0.0;
}

double PointSetTopologyContainer::getPZ(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][2];
    else
        return 0.0;
}

void PointSetTopologyContainer::init()
{
    core::topology::TopologyContainer::init();

    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if (nbPoints.getValue() == 0 && !initPoints.empty())
        nbPoints.setValue(initPoints.size());
}

void PointSetTopologyContainer::addPoints(const unsigned int nPoints)
{
    nbPoints.setValue( nbPoints.getValue() + nPoints);
}

void PointSetTopologyContainer::removePoints(const unsigned int nPoints)
{
    nbPoints.setValue(nbPoints.getValue() - nPoints);
}

void PointSetTopologyContainer::addPoint()
{
    nbPoints.setValue(nbPoints.getValue()+1);
}

void PointSetTopologyContainer::removePoint()
{
    nbPoints.setValue(nbPoints.getValue()-1);
}

#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
void PointSetTopologyContainer::updateTopologyEngineGraph()
{
    std::cout << "PointSetTopologyModifier::updateTopologyEngineGraph()" << std::endl;
}
#endif

} // namespace topology

} // namespace component

} // namespace sofa

