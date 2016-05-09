/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_MECHANICALVPRINTACTION_H
#define SOFA_SIMULATION_MECHANICALVPRINTACTION_H

#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <iostream>

#include <sofa/core/ExecParams.h>
#include <sofa/core/VecId.h>
#include <sofa/core/MultiVecId.h>


namespace sofa
{

namespace simulation
{


/** Print a vector */
class SOFA_SIMULATION_COMMON_API MechanicalVPrintVisitor : public Visitor
{
public:
    MechanicalVPrintVisitor(const core::ExecParams* params,
                            sofa::core::ConstMultiVecId v, std::ostream& =std::cerr );
    virtual Result processNodeTopDown(simulation::Node* node);

    virtual const char* getClassName() const { return "MechanicalVPrintVisitor"; }
protected:
    sofa::core::ConstMultiVecId v_;
    std::ostream& out_;
};


/** Print a vector with an elapsed time, useful to compare convergence in odeSolver in function of time */
class SOFA_SIMULATION_COMMON_API MechanicalVPrintWithElapsedTimeVisitor : public Visitor
{
protected:
    sofa::core::ConstMultiVecId v_;
    unsigned count_;
    unsigned time_;
    std::ostream& out_;
public:
    MechanicalVPrintWithElapsedTimeVisitor(const core::ExecParams* params,
                                           sofa::core::ConstMultiVecId vid, unsigned time, std::ostream& =std::cerr );
    virtual Result processNodeTopDown(simulation::Node* node);
    virtual const char* getClassName() const { return "MechanicalVPrintWithElapsedTimeVisitor"; }
};



class SOFA_SIMULATION_COMMON_API DofPrintVisitor : public Visitor
{
public:
    DofPrintVisitor(const core::ExecParams* params,
                    sofa::core::ConstMultiVecId v, const std::string& dofname, std::ostream& =std::cerr );
    virtual Result processNodeTopDown(simulation::Node* node);

    virtual const char* getClassName() const { return "DofPrintVisitor"; }
protected:
    sofa::core::ConstMultiVecId v_;
    std::ostream& out_;
    const std::string& dofname_;
};


} // namespace simulation

} // namespace sofa

#endif
