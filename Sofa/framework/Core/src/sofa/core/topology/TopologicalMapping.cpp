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
#include <sofa/core/topology/TopologicalMapping.h>

namespace sofa::core::topology
{

TopologicalMapping::TopologicalMapping()
    : fromModel(initLink("input", "Input topology to map"))
    , toModel(initLink("output", "Output topology to map"))
{

}


void TopologicalMapping::setTopologies(In* from, Out* to)
{
    this->fromModel.set(from);
    this->toModel.set(to);
}


Index TopologicalMapping::getGlobIndex(Index ind)
{
    if (ind < (Loc2GlobDataVec.getValue()).size())
    {
        return (Loc2GlobDataVec.getValue())[ind];
    }
    else
    {
        return 0;
    }
}

Index TopologicalMapping::getFromIndex(Index ind) 
{ 
    SOFA_UNUSED(ind);
    return 0; 
}


void TopologicalMapping::dumpGlob2LocMap()
{
    std::map<Index, Index>::iterator itM;
    msg_info() << "## Log Glob2LocMap - size: " << Glob2LocMap.size() << " ##";
    for (itM = Glob2LocMap.begin(); itM != Glob2LocMap.end(); ++itM)
        msg_info() << (*itM).first << " - " << (*itM).second;

    msg_info() << "#################";
}


void TopologicalMapping::dumpLoc2GlobVec()
{
    const sofa::type::vector<Index>& buffer = Loc2GlobDataVec.getValue();
    msg_info() << "## Log Loc2GlobDataVec - size: " << buffer.size() << " ##";
    for (Index i = 0; i < buffer.size(); ++i)
        msg_info() << i << " - " << buffer[i];

    msg_info() << "#################";
}


bool TopologicalMapping::checkTopologyInputTypes()
{
    if (m_inputType == geometry::ElementType::UNKNOWN)
    {
        dmsg_error() << "The input ElementType has not been set. Define 'm_inputType' to the correct ElementType in the constructor.";
        return false;
    }

    if (m_outputType == geometry::ElementType::UNKNOWN)
    {
        dmsg_error() << "The output ElementType has not been set. Define 'm_outputType' to the correct ElementType in the constructor.";
        return false;
    }

    assert(fromModel.get());
    assert(toModel.get());

    const ElementType inputTopologyType = fromModel->getTopologyType();
    if (inputTopologyType != m_inputType)
    {
        msg_error() << "The type of the input topology '" << fromModel.getPath() << "' (" << elementTypeToString(inputTopologyType) << ") does not correspond to a valid '" << elementTypeToString(m_inputType) << "' topology.";
        return false;
    }

    const ElementType outputTopologyType = toModel->getTopologyType();
    if (outputTopologyType != m_outputType)
    {
        msg_error() << "The type of the output topology '" << toModel.getPath() << "' (" << elementTypeToString(outputTopologyType) << ") does not correspond to a valid '" << elementTypeToString(m_outputType) << "' topology.";
        return false;
    }

    return true;
}

} /// namespace sofa::core::topology

