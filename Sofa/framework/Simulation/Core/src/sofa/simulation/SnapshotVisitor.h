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


#include <sofa/config.h>
#include <sofa/simulation/Visitor.h>
#include <string>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseSnapshot.h>
#include <stack>
#include <memory>


namespace sofa::simulation
{


class SOFA_SIMULATION_CORE_API SnapshotVisitor : public Visitor
{
protected:
    core::objectmodel::BaseSnapshot& snapCont_; 
    // core::objectmodel::BaseSnapshot::SnapNode& snapNode_;
    
public:
    SnapshotVisitor(const sofa::core::ExecParams* eparams, core::objectmodel::BaseSnapshot& snapshot) : Visitor(eparams), snapCont_(snapshot)
    {
    }

    void processObject(core::objectmodel::BaseObject* obj, std::string parentName);

    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* node) override;
    const char* getClassName() const override { return "SnapshotVisitor"; }

};

} // namespace sofa::simulation

