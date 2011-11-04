/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
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

    TopologyEngine() {}//m_topologicalData(NULL)  {}

    /*  virtual ~TopologyEngine()
      {
          if (this->m_topologicalData != NULL)
              this->removeTopologicalData();
      }
    */
    virtual void init()
    {
        // TODO: see if necessary or not....
        // this->addInput(&m_changeList);

        // TODO: understand why this crash!!
        //this->addOutput(this->m_topologicalData);

        this->createEngineName();
    }

    virtual void handleTopologyChange() {}


public:
    // really need to be a Data??
    Data <sofa::helper::list<const TopologyChange *> >m_changeList;

    unsigned int getNumberOfTopologicalChanges() {return (m_changeList.getValue()).size();}

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
            m_name = m_prefix + "no_name";
        else
            m_name = m_prefix + m_data_name;

        return;
    }

    const std::string& getName() const { return m_name; }

    void setName(const std::string& name) { m_name=name; }

    virtual void linkToPointDataArray() {}
    virtual void linkToEdgeDataArray() {}
    virtual void linkToTriangleDataArray() {}
    virtual void linkToQuadDataArray() {}
    virtual void linkToTetrahedronDataArray() {}
    virtual void linkToHexahedronDataArray() {}

protected:
    /// Data handle by the topological engine
    //t_topologicalData* m_topologicalData;

    //TopologyHandler* m_topologyHandler;

    /// Engine name base on Data handled: m_name = m_prefix+Data_name
    std::string m_name;
    /// use to define engine name.
    std::string m_prefix;
    /// use to define data handled name.
    std::string m_data_name;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H
