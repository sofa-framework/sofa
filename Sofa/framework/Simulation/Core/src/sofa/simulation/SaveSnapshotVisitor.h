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
#include <sofa/core/objectmodel/Base.h>
#include <sofa/simulation/Visitor.h>
#include <memory>

namespace sofa::simulation
{


class SOFA_SIMULATION_CORE_API SaveSnapshotVisitor : public Visitor
{
protected:
    core::objectmodel::Snapshot& m_snapshotContainer;

    std::unordered_map<core::objectmodel::BaseNode*, std::shared_ptr<core::objectmodel::Snapshot::SnapshotNode>> m_snapshotNodeMap;
    
public:
    SaveSnapshotVisitor(const sofa::core::ExecParams* eparams, core::objectmodel::Snapshot& snapshot) : Visitor(eparams), m_snapshotContainer(snapshot)
    {
    }

    void processObject(core::objectmodel::BaseObject* obj, std::shared_ptr<core::objectmodel::Snapshot::SnapshotNode> parent);

    Result processNodeTopDown(simulation::Node* node) override;
    const char* getClassName() const override { return "SaveSnapshotVisitor"; }

};

} // namespace sofa::simulation

