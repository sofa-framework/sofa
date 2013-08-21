/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_TREE_EXPORTINPACTION_H
#define SOFA_SIMULATION_TREE_EXPORTINPACTION_H

#include <sofa/core/ExecParams.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/misc/INPExporter.h>
#include <string>
#include <iostream>


namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_COMMON_API ExportINPVisitor : public Visitor
{
public:
    vector< std::string >* nameT;
    vector< defaulttype::Vec3Types::VecCoord >* positionT;
    vector< double >* densiT;
    vector< vector< sofa::component::topology::Tetrahedron > >* tetrahedraT;
    vector< vector< sofa::component::topology::Hexahedron > >* hexahedraT;
    vector< vector< unsigned int > >* fixedPointT;
    vector< double >* youngModulusT;
    vector< double >* poissonRatioT;

    ExportINPVisitor(const core::ExecParams* params /* PARAMS FIRST */, vector< std::string >* nameT, vector< defaulttype::Vec3Types::VecCoord >* positionT, vector< double >* densiT, vector< vector< sofa::component::topology::Tetrahedron > >* tetrahedraT, vector< vector< sofa::component::topology::Hexahedron > >* hexahedraT, vector< vector< unsigned int > >* fixedPointT, vector< double >* youngModulusT, vector< double >* poissonRatioT);
    ~ExportINPVisitor();

    virtual void processINPExporter(Node* node, core::exporter::BaseExporter* be);

    virtual Result processNodeTopDown(Node* node);
    virtual void processNodeBottomUp(Node* node);
    virtual const char* getClassName() const { return "ExportINPVisitor"; }

protected:
    int ID;
};

} // namespace simulation

} // namespace sofa

#endif // SOFA_SIMULATION_TREE_EXPORTINPACTION_H
