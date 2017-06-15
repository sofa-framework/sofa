/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_SIMULATION_TREE_EXPORTOBJACTION_H
#define SOFA_SIMULATION_TREE_EXPORTOBJACTION_H

#include <sofa/core/ExecParams.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/Node.h>
#include <string>
#include <iostream>


namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_CORE_API ExportOBJVisitor : public Visitor
{
public:
    std::ostream* out;
    std::ostream* mtl;

    ExportOBJVisitor(const core::ExecParams* params, std::ostream* out);
    ExportOBJVisitor(const core::ExecParams* params, std::ostream* out, std::ostream* mtl);
    ~ExportOBJVisitor();

    virtual void processVisualModel(Node* node, core::visual::VisualModel* vm);

    virtual Result processNodeTopDown(Node* node);
    virtual void processNodeBottomUp(Node* node);
    virtual const char* getClassName() const { return "ExportOBJVisitor"; }

protected:
    int ID;
    int vindex;
    int nindex;
    int tindex;
    int count;
};

} // namespace simulation

} // namespace sofa

#endif // SOFA_SIMULATION_TREE_EXPORTOBJACTION_H
