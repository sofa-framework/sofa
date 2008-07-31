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
#include <sofa/component/topology/PointSetTopologyModifier.h>
#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/PointSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
SOFA_DECL_CLASS(PointSetTopologyModifier)
int PointSetTopologyModifierClass = core::RegisterObject("Point set topology modifier")
        .add< PointSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;


void PointSetTopologyModifier::init()
{
    core::componentmodel::topology::TopologyModifier::init();
    this->getContext()->get(m_container);
}


void PointSetTopologyModifier::swapPoints(const int i1, const int i2)
{
    PointsIndicesSwap *e2 = new PointsIndicesSwap( i1, i2 );
    addStateChange(e2);
    m_container->propagateStateChanges();

    PointsIndicesSwap *e = new PointsIndicesSwap( i1, i2 );
    this->addTopologyChange(e);
}


void PointSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    m_container->addPoints(nPoints);
}

void PointSetTopologyModifier::addPointsWarning(const unsigned int nPoints, const bool addDOF)
{
    if(addDOF)
    {
        PointsAdded *e2 = new PointsAdded(nPoints);
        addStateChange(e2);
        m_container->propagateStateChanges();
    }

    // Warning that vertices just got created
    PointsAdded *e = new PointsAdded(nPoints);
    this->addTopologyChange(e);
}


void PointSetTopologyModifier::addPointsWarning(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double       > >& coefs,
        const bool addDOF)
{
    if(addDOF)
    {
        PointsAdded *e2 = new PointsAdded(nPoints, ancestors, coefs);
        addStateChange(e2);
        m_container->propagateStateChanges();
    }

    // Warning that vertices just got created
    PointsAdded *e = new PointsAdded(nPoints, ancestors, coefs);
    this->addTopologyChange(e);
}


void PointSetTopologyModifier::removePointsWarning(sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    // sort points so that they are removed in a descending order
    std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() );

    // Warning that these vertices will be deleted
    PointsRemoved *e = new PointsRemoved(indices);
    this->addTopologyChange(e);

    if(removeDOF)
    {
        PointsRemoved *e2 = new PointsRemoved(indices);
        addStateChange(e2);
    }
}


void PointSetTopologyModifier::removePointsProcess(const sofa::helper::vector<unsigned int> & indices,
        const bool removeDOF)
{
    if(removeDOF)
    {
        m_container->propagateStateChanges();
    }
    m_container->removePoints(indices.size());
}


void PointSetTopologyModifier::renumberPointsWarning( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    // Warning that these vertices will be deleted
    PointsRenumbering *e = new PointsRenumbering(index, inv_index);
    this->addTopologyChange(e);

    if(renumberDOF)
    {
        PointsRenumbering *e2 = new PointsRenumbering(index, inv_index);
        addStateChange(e2);
    }
}


void PointSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &/*index*/,
        const sofa::helper::vector<unsigned int> &/*inv_index*/,
        const bool renumberDOF)
{
    if(renumberDOF)
    {
        m_container->propagateStateChanges();
    }
}

} // namespace topology

} // namespace component

} // namespace sofa

