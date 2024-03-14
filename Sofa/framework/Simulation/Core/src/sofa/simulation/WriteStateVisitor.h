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
#ifndef SOFA_SIMULATION_WRITESTATEACTION_H
#define SOFA_SIMULATION_WRITESTATEACTION_H

#include <sofa/simulation/Visitor.h>


namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API WriteStateVisitor: public Visitor
{
public:
    WriteStateVisitor( const sofa::core::ExecParams* params, std::ostream& out );
    ~WriteStateVisitor() override;

    Result processNodeTopDown( simulation::Node*  ) override;
    const char* getClassName() const override { return "WriteStateVisitor"; }

protected:
    std::ostream& m_out;
};

} // namespace sofa::simulation


#endif
