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
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/TopologyHandler.h>

namespace sofa::core::topology
{

namespace
{
    constexpr sofa::Size getElementTypeIndex(sofa::geometry::ElementType elementType)
    {
        return static_cast<std::underlying_type_t<sofa::geometry::ElementType>>(elementType);
    }
}


// GeometryAlgorithms implementation

void GeometryAlgorithms::init()
{
}

void GeometryAlgorithms::initPointsAdded(const type::vector< sofa::Index >& /*indices*/, const type::vector< PointAncestorElem >& /*ancestorElems*/
    , const type::vector< core::VecCoordId >& /*coordVecs*/, const type::vector< core::VecDerivId >& /*derivVecs */)
{
}

// TopologyModifier implementation

void TopologyModifier::init()
{
    this->getContext()->get(m_topologyContainer);
}

void TopologyModifier::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addTopologyChange(topologyChange);
}

void TopologyModifier::addStateChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addStateChange(topologyChange);
}

void TopologyModifier::propagateStateChanges() {}
void TopologyModifier::propagateTopologicalChanges() {}
void TopologyModifier::notifyEndingEvent() {}
void TopologyModifier::removeItems(const sofa::type::vector< Index >& /*items*/) {}

// TopologyContainer implementation


TopologyContainer::~TopologyContainer()
{
    resetTopologyChangeList();
    resetStateChangeList();
    resetTopologyHandlerList();
}

void TopologyContainer::init()
{
    core::topology::BaseMeshTopology::init();
    core::topology::BaseTopologyObject::init();
}


void TopologyContainer::addTopologyChange(const TopologyChange *topologyChange)
{
    std::list<const TopologyChange *>& my_changeList = *(m_changeList.beginEdit());
    my_changeList.push_back(topologyChange);
    m_changeList.endEdit();
}

void TopologyContainer::addStateChange(const TopologyChange *topologyChange)
{
    std::list<const TopologyChange *>& my_stateChangeList = *(m_stateChangeList.beginEdit());
    my_stateChangeList.push_back(topologyChange);
    m_stateChangeList.endEdit();
}

void TopologyContainer::addTopologyHandler(TopologyHandler *_TopologyHandler, sofa::geometry::ElementType elementType)
{
    m_topologyHandlerListPerElement[getElementTypeIndex(elementType)].push_back(_TopologyHandler);
}

const std::list<TopologyHandler*>& TopologyContainer::getTopologyHandlerList(sofa::geometry::ElementType elementType) const
{
    return m_topologyHandlerListPerElement[getElementTypeIndex(elementType)];
}

void TopologyContainer::linkTopologyHandlerToData(TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    // default implementation dont do anything
    // as it does not have any data itself
    SOFA_UNUSED(topologyHandler);
    SOFA_UNUSED(elementType);
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::endChange() const
{
    return (m_changeList.getValue()).end();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::beginChange() const
{
    return (m_changeList.getValue()).begin();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::endStateChange() const
{
    return (m_stateChangeList.getValue()).end();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::beginStateChange() const
{
    return (m_stateChangeList.getValue()).begin();
}

void TopologyContainer::resetTopologyChangeList()
{
    std::list<const TopologyChange *>& my_changeList = *(m_changeList.beginEdit());
    for (std::list<const TopologyChange *>::iterator it=my_changeList.begin();
            it!=my_changeList.end(); ++it)
    {
        delete (*it);
    }

    my_changeList.clear();
    m_changeList.endEdit();
}

void TopologyContainer::resetStateChangeList()
{
    std::list<const TopologyChange *>& my_stateChangeList = *(m_stateChangeList.beginEdit());
    for (std::list<const TopologyChange *>::iterator it=my_stateChangeList.begin();
            it!=my_stateChangeList.end(); ++it)
    {
        delete (*it);
    }

    my_stateChangeList.clear();
    m_stateChangeList.endEdit();
}

void TopologyContainer::resetTopologyHandlerList()
{
    for (auto& topologyHandlerList : m_topologyHandlerListPerElement)
    {
        for (auto it = topologyHandlerList.begin();
            it != topologyHandlerList.end(); ++it)
        {
            *it = nullptr;
        }
        topologyHandlerList.clear();
    }
    m_topologyHandlerListPerElement.clear();
    m_topologyHandlerListPerElement.resize(sofa::geometry::NumberOfElementType);
}


void TopologyContainer::updateDataEngineGraph(const sofa::core::objectmodel::BaseData& my_Data, sofa::geometry::ElementType elementType)
{
    // clear data stored by previous call of this function
    m_topologyHandlerListPerElement[getElementTypeIndex(elementType)].clear();
    this->m_enginesGraph.clear();
    this->m_dataGraph.clear();


    sofa::core::objectmodel::DDGNode::DDGLinkContainer _outs = my_Data.getOutputs();
    sofa::core::objectmodel::DDGNode::DDGLinkIterator it;

    bool allDone = false;

    unsigned int cpt_security = 0;
    std::list<sofa::core::topology::TopologyHandler*> _engines;
    std::list<sofa::core::topology::TopologyHandler*>::iterator it_engines;

    while (!allDone && cpt_security < 1000)
    {
        std::list<sofa::core::objectmodel::DDGNode* > next_GraphLevel;
        std::list<sofa::core::topology::TopologyHandler*> next_enginesLevel;

        // for drawing graph
        sofa::type::vector<std::string> enginesNames;
        sofa::type::vector<std::string> dataNames;

        // doing one level of data outputs, looking for engines
        for (it = _outs.begin(); it != _outs.end(); ++it)
        {
            sofa::core::topology::TopologyHandler* topoEngine = dynamic_cast <sofa::core::topology::TopologyHandler*> ((*it));

            if (topoEngine)
            {
                next_enginesLevel.push_back(topoEngine);
                enginesNames.push_back(topoEngine->getName());
            }
        }

        _outs.clear();

        // looking for data linked to engines
        for (it_engines = next_enginesLevel.begin(); it_engines != next_enginesLevel.end(); ++it_engines)
        {
            // for each output engine, looking for data outputs

            // There is a conflict with Base::getOutputs()
            sofa::core::objectmodel::DDGNode* my_topoEngine = (*it_engines);
            const sofa::core::objectmodel::DDGNode::DDGLinkContainer& _outsTmp = my_topoEngine->getOutputs();
            sofa::core::objectmodel::DDGNode::DDGLinkIterator itTmp;

            for (itTmp = _outsTmp.begin(); itTmp != _outsTmp.end(); ++itTmp)
            {
                sofa::core::objectmodel::BaseData* data = dynamic_cast<sofa::core::objectmodel::BaseData*>((*itTmp));
                if (data)
                {
                    next_GraphLevel.push_back((*itTmp));
                    dataNames.push_back(data->getName());

                    const sofa::core::objectmodel::DDGNode::DDGLinkContainer& _outsTmp2 = data->getOutputs();
                    _outs.insert(_outs.end(), _outsTmp2.begin(), _outsTmp2.end());
                }
            }

            this->m_dataGraph.push_back(dataNames);
            dataNames.clear();
        }


        // Iterate:
        _engines.insert(_engines.end(), next_enginesLevel.begin(), next_enginesLevel.end());
        this->m_enginesGraph.push_back(enginesNames);

        if (next_GraphLevel.empty()) // end
            allDone = true;

        next_GraphLevel.clear();
        next_enginesLevel.clear();
        enginesNames.clear();

        cpt_security++;
    }


    // check good loop escape
    if (cpt_security >= 1000)
        msg_error() << "PointSetTopologyContainer::updateTopologyHandlerGraph reach end loop security.";


    // Reorder engine graph by inverting order and avoiding duplicate engines
    std::list<sofa::core::topology::TopologyHandler*>::reverse_iterator it_engines_rev;

    for(auto& topologyHandlerList : m_topologyHandlerListPerElement)
    {
        for (it_engines_rev = _engines.rbegin(); it_engines_rev != _engines.rend(); ++it_engines_rev)
        {
            bool find = false;

            for (it_engines = topologyHandlerList.begin(); it_engines != topologyHandlerList.end(); ++it_engines)
            {
                if ((*it_engines_rev) == (*it_engines))
                {
                    find = true;
                    break;
                }
            }

            if (!find)
                topologyHandlerList.push_back((*it_engines_rev));
        }
    }

    return;
}

} // namespace sofa::core::topology
