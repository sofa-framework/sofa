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
#include <sofa/simulation/MappingGraph.h>
#include <sofa/simulation/mappinggraph/ExportDot.h>

#include <sstream>

namespace sofa::simulation
{

std::string getType(BaseMappingGraphNode* node)
{
    const auto type = node->getType();
    if (type == BaseMappingGraphNode::NodeType::Mapping)
        return "Mapping";
    else if (type == BaseMappingGraphNode::NodeType::Component)
        return "Component";
    else if (type == BaseMappingGraphNode::NodeType::MechanicalState)
        return "State";
    else if (type == BaseMappingGraphNode::NodeType::Group)
        return "Group";
    else
        return "Unknown";
}

std::string getColor(BaseMappingGraphNode* node)
{
    const auto type = node->getType();
    if (type == BaseMappingGraphNode::NodeType::Mapping)
        return "red1";
    else if (type == BaseMappingGraphNode::NodeType::Component)
        return "royalblue1";
    else if (type == BaseMappingGraphNode::NodeType::MechanicalState)
        return "seagreen1";
    else if (type == BaseMappingGraphNode::NodeType::Group)
        return "slateblue1";
    else
        return "webgray";
}

std::string getShape(BaseMappingGraphNode* node)
{
    const auto type = node->getType();
    if (type == BaseMappingGraphNode::NodeType::Mapping)
        return "diamond";
    else if (type == BaseMappingGraphNode::NodeType::Component)
        return "ellipse";
    else if (type == BaseMappingGraphNode::NodeType::MechanicalState)
        return "box";
    else if (type == BaseMappingGraphNode::NodeType::Group)
        return "component";
    else
        return "note";
}

std::string exportToDotFormat(const MappingGraph& graph)
{
    // Use a string stream to build the DOT content
    std::stringstream ss;
    ss << "digraph  MappingGraph {\n";
    ss << "  rankdir=TB;\n"; // Top to Bottom layout is common for dependency graphs

    // 1. Add all nodes (vertices)
    for (const auto& node : graph.getAllNodes())
    {
        const std::string label = "[" + getType(node.get()) + "]" + node->getName();

        // Assuming a unique name or ID can be generated for the node in DOT format
        ss << "  \"Node_" << std::to_string(reinterpret_cast<uintptr_t>(node.get()))
            << "\" [label=\""<< label << "\", color=\"" << getColor(node.get()) << "\", shape=\""
            << getShape(node.get()) << "\"];\n";
    }

    // 2. Add all edges (dependencies)
    for (const auto& node : graph.getAllNodes())
    {
        const std::string sourceName = "Node_" + std::to_string(reinterpret_cast<uintptr_t>(node.get()));
        for (const auto& child : node->getChildren())
        {
            const std::string targetName = "Node_" + std::to_string(reinterpret_cast<uintptr_t>(child.get()));

            // Assuming edges represent dependencies (Source -> Target)
            ss << "  " << sourceName << " -> " << targetName << ";\n";
        }
    }

    ss << "}\n";
    return ss.str();
}

}  // namespace sofa::simulation
