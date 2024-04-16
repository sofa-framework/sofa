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
#include <sofa/component/io/mesh/VisualModelOBJExporter.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/ExportVisualModelOBJVisitor.h>
using sofa::simulation::ExportVisualModelOBJVisitor ;

namespace sofa::component::_visualmodelobjexporter_
{

int VisualModelOBJExporterClass = core::RegisterObject("Export the scene under the Wavefront OBJ format."
                                            "When several frames are exported the file name have the following pattern: outfile000.obj outfile001.obj.")
        .add< VisualModelOBJExporter >()
        .addAlias("ObjExporter")
        .addAlias("OBJExporter");


VisualModelOBJExporter::~VisualModelOBJExporter()
{
}


bool VisualModelOBJExporter::write()
{
    return writeOBJ() ;
}


bool VisualModelOBJExporter::writeOBJ()
{
    std::string basename = getOrCreateTargetPath(d_filename.getValue(),
                                                 d_exportEveryNbSteps.getValue()) ;
    std::string objfilename = basename ;
    std::string mtlfilename = basename ;

    if ( !(objfilename.size() > 3 && objfilename.substr(objfilename.size()-4)==".obj"))
        objfilename += ".obj";
    std::ofstream outfile(objfilename.c_str());

    if ( !(mtlfilename.size() > 3 && mtlfilename.substr(objfilename.size()-4)==".obj"))
        mtlfilename += ".mtl";
    else
        mtlfilename = mtlfilename.substr(0, mtlfilename.size()-4) + ".mtl";
    std::ofstream mtlfile(mtlfilename.c_str());

    if(!outfile.is_open())
    {
        msg_warning() << "Unable to export OBJ...the file '"<< objfilename <<"' cannot be opened" ;
        return false ;
    }

    if(!mtlfile.is_open())
    {
        msg_warning() << "Unable to export OBJ...the file '"<< objfilename <<"' cannot be opened" ;
        return false ;
    }

    ExportVisualModelOBJVisitor exportOBJ(core::execparams::defaultInstance(),&outfile, &mtlfile);
    getContext()->executeVisitor(&exportOBJ);

    outfile.close();
    mtlfile.close();

    msg_info() << "Exporting OBJ in: " << objfilename.c_str() << " with MTL in: " << mtlfilename.c_str() ;
    return true ;
}


void VisualModelOBJExporter::handleEvent(Event *event)
{
    BaseSimulationExporter::handleEvent(event) ;
}

} // namespace sofa::component::_visualmodelobjexporter_
