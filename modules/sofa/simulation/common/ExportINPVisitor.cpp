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
#include <sofa/simulation/common/ExportINPVisitor.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/objectmodel/BaseContext.h>
namespace sofa
{

namespace simulation
{


    ExportINPVisitor::ExportINPVisitor(const core::ExecParams* params /* PARAMS FIRST */, vector< std::string >* nameT, vector< defaulttype::Vec3Types::VecCoord >* positionT, vector< double >* densiT, vector< vector< sofa::component::topology::Tetrahedron > >* tetrahedraT, vector< vector< sofa::component::topology::Hexahedron > >* hexahedraT, vector< vector< unsigned int > >* fixedPointT, vector< double >* youngModulusT, vector< double >* poissonRatioT)
    : Visitor(params), nameT(nameT), positionT(positionT), densiT(densiT), tetrahedraT(tetrahedraT), hexahedraT(hexahedraT), fixedPointT(fixedPointT), youngModulusT(youngModulusT), poissonRatioT(poissonRatioT), ID(0)
{
}

ExportINPVisitor::~ExportINPVisitor()
{
}

void ExportINPVisitor::processINPExporter(Node* /*node*/, core::exporter::BaseExporter* be)
{
    std::ostringstream oname;
    oname << ++ID << "_" << be->getName() << std::endl;
    if(!be->getINP(nameT, positionT, densiT, tetrahedraT, hexahedraT, fixedPointT, youngModulusT, poissonRatioT))
    {
        std::cout << "ExportINPVisitor: error, failed to get INPExporter component " << std::endl;
        return;
    }
}

simulation::Visitor::Result ExportINPVisitor::processNodeTopDown(Node* node)
{
    for_each(this, node, node->exporter,&ExportINPVisitor::processINPExporter);

    return RESULT_CONTINUE;
}

void ExportINPVisitor::processNodeBottomUp(Node* /*node*/)
{
}

} // namespace simulation

} // namespace sofa

