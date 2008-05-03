/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/tree/ExportOBJVisitor.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

ExportOBJVisitor::ExportOBJVisitor(std::ostream* out,std::ostream* mtl)
    : out(out), mtl(mtl), vindex(0), nindex(0), tindex(0)
{
}

ExportOBJVisitor::~ExportOBJVisitor()
{
}

void ExportOBJVisitor::processVisualModel(GNode* node, core::VisualModel* vm)
{
// 	GL::OglModel* oglmodel = dynamic_cast<GL::OglModel*>(vm);
// 	if (oglmodel != NULL)
// 	{
// 		std::string name = node->getPathName() + "/" + oglmodel->getName();

    std::string name;
    if( node->parent ) name += node->parent->getName() + "_";
    name += vm->getName();

// 	name += oglmodel->getName();


    // 		*out << "g "<<name<<"\n";
    //oglmodel->exportOBJ(out,mtl,vindex,nindex,tindex); // does not compile
// 	oglmodel->exportOBJ("Which-string-here_?",out,mtl,vindex,nindex,tindex); // changed by FF


    vm->exportOBJ(name,out,mtl,vindex,nindex,tindex);
// 	}

}

Visitor::Result ExportOBJVisitor::processNodeTopDown(GNode* node)
{
    //simulation::Node* node = static_cast<simulation::Node*>(n);
    for_each(this, node, node->visualModel, &ExportOBJVisitor::processVisualModel);

    return RESULT_CONTINUE;
}

void ExportOBJVisitor::processNodeBottomUp(GNode* /*node*/)
{
}

} // namespace tree

} // namespace simulation

} // namespace sofa

