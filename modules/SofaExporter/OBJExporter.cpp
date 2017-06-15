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

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(OBJExporter)

int OBJExporterClass = core::RegisterObject("Export under Wavefront OBJ format")
        .add< OBJExporter >()
        .addAlias("ObjExporter");

OBJExporter::OBJExporter()
    : stepCounter(0)
    , objFilename( initData(&objFilename, "filename", "output OBJ file name"))
    , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable)"))
    , exportAtBegin( initData(&exportAtBegin, false, "exportAtBegin", "export file at the initialization"))
    , exportAtEnd( initData(&exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished"))
    , activateExport(false)
{
    this->f_listening.setValue(true);
}

OBJExporter::~OBJExporter()
{
}

void OBJExporter::init()
{
    context = this->getContext();
    maxStep = exportEveryNbSteps.getValue();
}

void OBJExporter::writeOBJ()
{
    std::string filename = objFilename.getFullPath();
    if (maxStep)
    {
        std::ostringstream oss;
        oss.width(5);
        oss.fill('0');
        oss << stepCounter / maxStep;
        filename += oss.str();
    }
    if ( !(filename.size() > 3 && filename.substr(filename.size()-4)==".obj"))
        filename += ".obj";
    std::ofstream outfile(filename.c_str());

    std::string mtlfilename = objFilename.getFullPath();
    if ( !(mtlfilename.size() > 3 && mtlfilename.substr(filename.size()-4)==".obj"))
        mtlfilename += ".mtl";
    else
        mtlfilename = mtlfilename.substr(0, mtlfilename.size()-4) + ".mtl";
    std::ofstream mtlfile(mtlfilename.c_str());
    sofa::simulation::ExportOBJVisitor exportOBJ(core::ExecParams::defaultInstance(),&outfile, &mtlfile);
    context->executeVisitor(&exportOBJ);
    outfile.close();
    mtlfile.close();

    msg_info() << "Exporting OBJ as: " << filename.c_str() << " with MTL file: " << mtlfilename.c_str() ;
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
            writeOBJ();
            break;
        }

        case 'P':
        case 'p':
        {
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
