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
#ifndef SOFA_SIMULATION_MECHANICALVPRINTACTION_H
#define SOFA_SIMULATION_MECHANICALVPRINTACTION_H

#include <sofa/simulation/Visitor.h>
#include <iostream>


#include <sofa/core/MultiVecId.h>


namespace sofa::simulation
{


/** Print a vector */
class SOFA_SIMULATION_CORE_API MechanicalVPrintVisitor : public Visitor
{
public:
    MechanicalVPrintVisitor(const core::ExecParams* params,
                            sofa::core::ConstMultiVecId v, std::ostream& =std::cerr );
    Result processNodeTopDown(simulation::Node* node) override;

    const char* getClassName() const override { return "MechanicalVPrintVisitor"; }
protected:
    sofa::core::ConstMultiVecId v_;
    std::ostream& out_;
};


/** Print a vector with an elapsed time, useful to compare convergence in odeSolver in function of time */
class SOFA_SIMULATION_CORE_API MechanicalVPrintWithElapsedTimeVisitor : public Visitor
{
protected:
    sofa::core::ConstMultiVecId v_;
    unsigned count_;
    unsigned time_;
    std::ostream& out_;
public:
    MechanicalVPrintWithElapsedTimeVisitor(const core::ExecParams* params,
                                           sofa::core::ConstMultiVecId vid, unsigned time, std::ostream& =std::cerr );
    Result processNodeTopDown(simulation::Node* node) override;
    const char* getClassName() const override { return "MechanicalVPrintWithElapsedTimeVisitor"; }
};



class SOFA_SIMULATION_CORE_API DofPrintVisitor : public Visitor
{
public:
    DofPrintVisitor(const core::ExecParams* params,
                    sofa::core::ConstMultiVecId v, const std::string& dofname, std::ostream& =std::cerr );
    Result processNodeTopDown(simulation::Node* node) override;

    const char* getClassName() const override { return "DofPrintVisitor"; }
protected:
    sofa::core::ConstMultiVecId v_;
    std::ostream& out_;
    const std::string& dofname_;
};


} // namespace sofa::simulation


#endif
