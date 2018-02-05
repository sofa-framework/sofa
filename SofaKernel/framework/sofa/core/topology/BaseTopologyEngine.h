/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H
#define SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H

#include <sofa/core/topology/TopologyChange.h>

namespace sofa
{

namespace core
{

namespace topology
{



/** A class that will interact on a topological Data */
class TopologyEngine : public sofa::core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(TopologyEngine, DataEngine);
    //typedef sofa::core::objectmodel::Data< sofa::helper::vector <void*> > t_topologicalData;

protected:
    TopologyEngine() {}//m_topologicalData(NULL)  {}

    virtual ~TopologyEngine()
    {
        //if (this->m_topologicalData != NULL)
        //    this->removeTopologicalData();
    }

public:

    virtual void init() override
    {
        sofa::core::DataEngine::init();
        // TODO: see if necessary or not....
        // this->addInput(&m_changeList);

        // TODO: understand why this crash!!
        //this->addOutput(this->m_topologicalData);

        this->createEngineName();
    }

    virtual void handleTopologyChange() override {}


public:
    // really need to be a Data??
    Data <std::list<const TopologyChange *> >m_changeList;

    size_t getNumberOfTopologicalChanges() {return (m_changeList.getValue()).size();}

    //virtual void registerTopologicalData(t_topologicalData* topologicalData) {m_topologicalData = topologicalData;}
    /*
        virtual void removeTopologicalData()
        {
            if (this->m_topologicalData)
                delete this->m_topologicalData;
        }
    */
    //virtual const t_topologicalData* getTopologicalData() {return m_topologicalData;}

    virtual void createEngineName()
    {
        if (m_data_name.empty())
            setName( m_prefix + "no_name" );
        else
            setName( m_prefix + m_data_name );

        return;
    }

    virtual void linkToPointDataArray() {}
    virtual void linkToEdgeDataArray() {}
    virtual void linkToTriangleDataArray() {}
    virtual void linkToQuadDataArray() {}
    virtual void linkToTetrahedronDataArray() {}
    virtual void linkToHexahedronDataArray() {}

    void setNamePrefix(const std::string& s) { m_prefix = s; }

protected:
    /// Data handle by the topological engine
    //t_topologicalData* m_topologicalData;

    //TopologyHandler* m_topologyHandler;

    /// use to define engine name.
    std::string m_prefix;
    /// use to define data handled name.
    std::string m_data_name;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H
