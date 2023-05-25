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
#include <sofa/simulation/ExportXML.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/XMLPrintVisitor.h>

#include <fstream>

namespace sofa::simulation
{

void exportNodeInXML(sofa::simulation::Node* root, const char* fileName)
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    if ( fileName!=nullptr )
    {
        std::ofstream out ( fileName );
        out << "<?xml version=\"1.0\"?>\n";

        XMLPrintVisitor print ( params, out );
        root->execute ( print );
    }
    else
    {
        XMLPrintVisitor print ( params, std::cout );
        root->execute ( print );
    }
}

}
