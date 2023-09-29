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
#include <sofa/simulation/BaseSimulationExporter.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/events/SimulationInitDoneEvent.h>
#include <sofa/helper/system/FileSystem.h>

namespace sofa
{

namespace simulation
{

namespace _basesimulationexporter_
{
using sofa::simulation::SimulationInitDoneEvent ;
using sofa::simulation::AnimateEndEvent ;
using sofa::helper::system::FileSystem ;


BaseSimulationExporter::BaseSimulationExporter() :
    d_filename(initData(&d_filename, "filename",
                        "Path or filename where to export the data.  If missing the name of the component is used."))
  , d_exportEveryNbSteps(initData(&d_exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps",
                                  "export file only at specified number of steps (0=disable, default=0)"))
  , d_exportAtBegin( initData(&d_exportAtBegin, false, "exportAtBegin",
                              "export file before the simulation starts, once the simulation is initialized (default=false)"))
  , d_exportAtEnd( initData(&d_exportAtEnd, false, "exportAtEnd",
                            "export file when the simulation is over and cleanup is called, i.e. just before deleting the simulation (default=false)"))
  , d_isEnabled( initData(&d_isEnabled, true, "enable", "Enable or disable the component. (default=true)"))
{
    f_listening.setValue(false) ;
    d_filename.setPathType(sofa::core::objectmodel::PathType::BOTH);
}


const std::string BaseSimulationExporter::getOrCreateTargetPath(const std::string& filename, bool autonumbering)
{
    std::string path = FileSystem::cleanPath(filename) ;
    if( FileSystem::exists(path) && FileSystem::isDirectory(path) )
    {
        path = FileSystem::append(path, getName());
    }

    /// If the path does not exists on the FS...we create It
    const std::string parentPath = FileSystem::getParentDirectory(path) ;
    if( !FileSystem::exists(parentPath) ){
        FileSystem::findOrCreateAValidPath(parentPath) ;
    }

    /// At this point we have a valid path. We can now add a number indicating the frame save.
    if (autonumbering)
    {
        std::ostringstream oss;
        oss.width(5);
        oss.fill('0');
        oss << m_stepCounter / d_exportEveryNbSteps.getValue();
        path += oss.str();
    }
    return path ;
}

void BaseSimulationExporter::handleEvent(Event *event){
    if (AnimateEndEvent::checkEventType(event))
    {
        if(d_isEnabled.getValue()) {
            const auto maxStep = d_exportEveryNbSteps.getValue() ;

            if (maxStep == 0)
                return;

            m_stepCounter++;
            if(m_stepCounter % maxStep == 0)
            {
                write();
            }
        }
    }
    else if (SimulationInitDoneEvent::checkEventType(event))
    {
        if (d_isEnabled.getValue() && d_exportAtBegin.getValue())
        {
            write();
        }
    }

    BaseObject::handleEvent(event) ;
}


void BaseSimulationExporter::init()
{
    updateFromDataField() ;
    doInit() ;
}


void BaseSimulationExporter::reinit()
{
    updateFromDataField() ;
    doReInit();
}

void BaseSimulationExporter::updateFromDataField()
{
    /// We need to set a default filename... So which one ?
    if(!d_filename.isSet() || d_filename.getValue().empty())
    {
        d_filename.setValue(getName());
    }

    /// Activate the listening to the event in order to be able to export file once initialization is done or the nth-step
    if(d_exportEveryNbSteps.getValue() != 0 || d_exportAtBegin.getValue())
        this->f_listening.setValue(true);
}


void BaseSimulationExporter::cleanup()
{
    if (d_isEnabled.getValue() && d_exportAtEnd.getValue())
        write();
}


} /// namespace _baseexporter_

} /// namespace core

} /// namespace sofa
