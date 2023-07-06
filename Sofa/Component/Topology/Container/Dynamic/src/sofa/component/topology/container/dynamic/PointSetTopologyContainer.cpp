/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>

#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <algorithm>

namespace sofa::component::topology::container::dynamic
{

namespace
{

struct GeneratePointID
{
    typedef sofa::core::topology::BaseMeshTopology::PointID PointID;

    GeneratePointID( PointID startId = PointID(0) )
    :current(startId)
    {
    }

    PointID operator() () { return current++; }

    PointID current;
};

}

using namespace sofa::defaulttype;

int PointSetTopologyContainerClass = core::RegisterObject("Point set topology container")
        .add< PointSetTopologyContainer >()
        ;

PointSetTopologyContainer::PointSetTopologyContainer(Size npoints)
    : d_initPoints (initData(&d_initPoints, "position", "Initial position of points",true,true))
    , d_checkTopology (initData(&d_checkTopology, false, "checkTopology", "Parameter to activate internal topology checks (might slow down the simulation)"))
    , nbPoints (initData(&nbPoints, npoints, "nbPoints", "Number of points"))
{
    addAlias(&d_initPoints,"points");
}

void PointSetTopologyContainer::setNbPoints(Size n)
{
    nbPoints.setValue(n);  
}

Size PointSetTopologyContainer::getNumberOfElements() const
{
    return nbPoints.getValue();
}

bool PointSetTopologyContainer::checkTopology() const
{
    return true;
}

void PointSetTopologyContainer::clear()
{
    nbPoints.setValue(sofa::Size(0));
    helper::WriteAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    initPoints.clear();
}

void PointSetTopologyContainer::addPoint(SReal px, SReal py, SReal pz)
{
    // NB: This implementation of addPoint was and is still very dangerous to use since it compromises any prior 
    // modifications that were done on the container. The new size is imposed by the size of the initPoints array,
    // which is not maintained whatsoever by the other add / remove point methods.

    auto initPoints = sofa::helper::getWriteAccessor(d_initPoints);
    initPoints.push_back(InitTypes::Coord(px, py, pz));
    if (initPoints.size() > nbPoints.getValue())
    {
        setNbPoints(Size(initPoints.size()));
    }
}

bool PointSetTopologyContainer::hasPos() const
{
    const helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    return !initPoints.empty();
}

SReal PointSetTopologyContainer::getPX(Index i) const
{
    const helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][0];
    else
        return 0.0;
}

SReal PointSetTopologyContainer::getPY(Index i) const
{
    const helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][1];
    else
        return 0.0;
}

SReal PointSetTopologyContainer::getPZ(Index i) const
{
    const helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][2];
    else
        return 0.0;
}

void PointSetTopologyContainer::init()
{
    core::topology::TopologyContainer::init();
    const helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    const int pointsDiff = (int)initPoints.size() - (int)getNbPoints(); 
    if( pointsDiff > 0 )
    {
        addPoints( pointsDiff );
    }

}

void PointSetTopologyContainer::addPoints(const Size nPoints)
{
    setNbPoints( nbPoints.getValue() + nPoints );
}

void PointSetTopologyContainer::removePoints(const Size nPoints)
{
    setNbPoints( nbPoints.getValue() - nPoints );
}

void PointSetTopologyContainer::addPoint()
{
    setNbPoints( nbPoints.getValue() + 1 );
}

void PointSetTopologyContainer::removePoint()
{
    //nbPoints.setValue(nbPoints.getValue()-1);
    setNbPoints( nbPoints.getValue() - 1 );
}

void PointSetTopologyContainer::setPointTopologyToDirty()
{
    // set this container to dirty
    m_pointTopologyDirty = true;

    // set all engines link to this container to dirty
    auto& pointTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::POINT);
    for (const auto topoHandler : pointTopologyHandlerList)
    {
        topoHandler->setDirtyValue();
        msg_info() << "Point Topology Set dirty engine: " << topoHandler->getName();
    }
}

void PointSetTopologyContainer::cleanPointTopologyFromDirty()
{
    m_pointTopologyDirty = false;

    // security, clean all engines to avoid loops
    auto& pointTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::POINT);
    for (const auto topoHandler : pointTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            msg_warning() << "Point Topology update did not clean engine: " << topoHandler->getName();
            topoHandler->cleanDirty();
        }
    }
}

bool PointSetTopologyContainer::linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::POINT)
    {
        d_initPoints.addOutput(topologyHandler);
        return true;
    }
    else
    {
        return false;
    }
}


bool PointSetTopologyContainer::unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::POINT)
    {
        d_initPoints.delOutput(topologyHandler);
        return true;
    }
    else
    {
        return false;
    }
}


} //namespace sofa::component::topology::container::dynamic
