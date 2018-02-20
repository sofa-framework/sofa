/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
 *  Contributors:
 *       - froy
 *       - damien.marchal@univ-lille1.fr
 ***********************************************************************************/

#include "OBJExporter.h"

#include <sstream>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/ExportOBJVisitor.h>
using sofa::simulation::ExportOBJVisitor ;

#include <sofa/core/objectmodel/KeypressedEvent.h>
using sofa::core::objectmodel::KeypressedEvent ;

#include <sofa/core/objectmodel/KeyreleasedEvent.h>
using sofa::core::objectmodel::KeyreleasedEvent ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

namespace sofa
{

namespace component
{

namespace _objexporter_
{

SOFA_DECL_CLASS(OBJExporter)

int OBJExporterClass = core::RegisterObject("Export the scene under the Wavefront OBJ format."
                                            "When several frames are exported the file name have the following pattern: outfile000.obj outfile001.obj.")
        .add< OBJExporter >()
        .addAlias("ObjExporter");


OBJExporter::~OBJExporter()
{
}


bool OBJExporter::write()
{
    return writeOBJ() ;
}


bool OBJExporter::writeOBJ()
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

    ExportOBJVisitor exportOBJ(core::ExecParams::defaultInstance(),&outfile, &mtlfile);
    getContext()->executeVisitor(&exportOBJ);

    outfile.close();
    mtlfile.close();

    msg_info() << "Exporting OBJ in: " << objfilename.c_str() << " with MTL in: " << mtlfilename.c_str() ;
    return true ;
}


void OBJExporter::handleEvent(Event *event)
{
    if (KeypressedEvent::checkEventType(event))
    {
        KeypressedEvent *ev = static_cast<KeypressedEvent *>(event);

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
            //todo(18.06) remove the behavior
            msg_deprecated() << "Hard coded interaction behavior in component is now a deprecated behavior."
                                "Scene specific interaction should be implement using an external controller or pythonScriptController"
                                "Please update your scene because this behavior will be removed in Sofa 18.06";

            if (!d_isEnabled.getValue()){
                msg_info() << "Starting OBJ sequence export..." ;
            }else{
                msg_info() << "Ending OBJ sequence export..." ;
            }
            d_isEnabled = !d_isEnabled.getValue() ;
            break;
        }
        }
    }

    BaseSimulationExporter::handleEvent(event) ;
}

}

}

}
