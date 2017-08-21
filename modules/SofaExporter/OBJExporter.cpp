/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
/*
 * OBJExporter.cpp
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#include "OBJExporter.h"

#include <sstream>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/ExportOBJVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(OBJExporter)

int OBJExporterClass = core::RegisterObject("Export the scene under the Wavefront OBJ format."
                                            "When several frames are exported the file name have the following pattern: outfile000.obj outfile001.obj.")
        .add< OBJExporter >()
        .addAlias("ObjExporter");

OBJExporter::OBJExporter()
    : stepCounter(0)
    , objFilename( initData(&objFilename, "filename", "output OBJ file name. If missing the name of the component is used."))
    , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable, default=0)"))
    , exportAtBegin( initData(&exportAtBegin, false, "exportAtBegin", "export file at the initialization (default=false)"))
    , exportAtEnd( initData(&exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished (default=false)"))
    , activateExport(false)
{
    f_listening.setValue(true) ;
}

OBJExporter::~OBJExporter()
{
}

void OBJExporter::init()
{
    context = this->getContext();
    maxStep = exportEveryNbSteps.getValue();

    /// We need to set a default filename... So which one ?
    if(!objFilename.isSet() || objFilename.getValue().empty())
    {
        objFilename.setValue(getName());
    }
}

std::string findOrCreateAValidPath(const std::string path)
{
    if( FileSystem::exists(path) )
        return path ;

    std::string parentPath = FileSystem::getParentDirectory(path) ;
    std::string currentFile = FileSystem::stripDirectory(path) ;
    FileSystem::createDirectory(findOrCreateAValidPath( parentPath )+"/"+currentFile) ;
    return path ;
}

bool OBJExporter::writeOBJ()
{
    std::string path = FileSystem::cleanPath(objFilename.getFullPath()) ;
    if( FileSystem::exists(path) && FileSystem::isDirectory(path) ){
        path += "/" + getName() ;
    }
    /// If path does not exists...we create It
    std::string parentPath = FileSystem::getParentDirectory(path) ;
    if( !FileSystem::exists(parentPath) ){
        findOrCreateAValidPath(parentPath) ;
    }

    std::string objfilename = path ;
    std::string mtlfilename = path;

    if (maxStep)
    {
        std::ostringstream oss;
        oss.width(5);
        oss.fill('0');
        oss << stepCounter / maxStep;
        objfilename += oss.str();
    }
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

    sofa::simulation::ExportOBJVisitor exportOBJ(core::ExecParams::defaultInstance(),&outfile, &mtlfile);
    context->executeVisitor(&exportOBJ);

    outfile.close();
    mtlfile.close();

    msg_info() << "Exporting OBJ in: " << objfilename.c_str() << " with MTL in: " << mtlfilename.c_str() ;
    return true ;
}

void OBJExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);

        switch(ev->getKey())
        {

        case 'E':
        case 'e':
        {
            //todo(18.06) remove the behavior
            msg_deprecated() << "Hard coded interaction behavior in component is now a deprecated behavior."
                                "Scene specific interaction should be implement using an external controller or pythonScriptController."
                                "Please update your scene because this behavior will be removed in Sofa 18.06";
            writeOBJ();
            break;
        }

        case 'P':
        case 'p':
        {
            //todo(18.06)
            msg_deprecated() << "Hard coded interaction behavior in component is now a deprecated behavior."
                                "Scene specific interaction should be implement using an external controller or pythonScriptController"
                                "Please update your scene because this behavior will be removed in Sofa 18.06";

            if (!activateExport){
                msg_info() << "Starting OBJ sequence export..." ;
            }else{
                msg_info() << "Ending OBJ sequence export..." ;
            }
            activateExport = !activateExport;
            break;
        }
        }
    }

    if ( simulation::AnimateEndEvent::checkEventType(event))
    {
        if (maxStep == 0 || !activateExport) return;

        stepCounter++;
        if(stepCounter % maxStep == 0)
        {
            writeOBJ();
        }
    }
}

void OBJExporter::cleanup()
{
    if (exportAtEnd.getValue())
        writeOBJ();
}

void OBJExporter::bwdInit()
{
    if (exportAtBegin.getValue())
        writeOBJ();
}

}

}

}
