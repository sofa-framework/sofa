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
#ifndef SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H
#define SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/fwd.h>
#include <sofa/helper/list.h>

#ifndef SOFA_CORE_TOPOLOGY_BASETOPOLOGYENGINE_DEFINITION
namespace std
{
    extern template class list<const sofa::core::topology::TopologyChange*>;
}
namespace sofa::core::objectmodel
{
    extern template class Data<std::list<const sofa::core::topology::TopologyChange*>>;
}

#endif /// SOFA_CORE_TOPOLOGY_BASETOPOLOGYENGINE_DEFINITION

namespace sofa
{

namespace core
{

namespace topology
{

/** A class that will interact on a topological Data */
class SOFA_CORE_API TopologyEngine : public sofa::core::DataEngine
{
public:
    SOFA_ABSTRACT_CLASS(TopologyEngine, DataEngine);

protected:
    TopologyEngine() {}
    ~TopologyEngine() override {}

public:
    void init() override ;
    void handleTopologyChange() override {}

public:
    // really need to be a Data??
    Data <std::list<const TopologyChange *> >m_changeList;

    size_t getNumberOfTopologicalChanges();

    virtual void createEngineName();
    virtual void linkToPointDataArray() {}
    virtual void linkToEdgeDataArray() {}
    virtual void linkToTriangleDataArray() {}
    virtual void linkToQuadDataArray() {}
    virtual void linkToTetrahedronDataArray() {}
    virtual void linkToHexahedronDataArray() {}

    void setNamePrefix(const std::string& s) { m_prefix = s; }

protected:
    /// use to define engine name.
    std::string m_prefix;
    /// use to define data handled name.
    std::string m_data_name;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_BASETOPOLOGYENGINE_H
