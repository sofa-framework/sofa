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
/******************************************************************************
 *  Contributors:
 *    - damien.marchal@univ-lille1.fr
 *****************************************************************************/
#ifndef SOFA_CORE_EXPORTER_BASEEXPORTER_H
#define SOFA_CORE_EXPORTER_BASEEXPORTER_H

#include <sofa/simulation/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <string>

namespace sofa
{

namespace simulation
{

namespace _basesimulationexporter_
{
using sofa::core::objectmodel::Event ;
using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::DataFileName ;

/**
    Componant that export something from the scene could inherit from this class
    as it implement an uniform handling of the differen data attributes.
*/
class SOFA_SIMULATION_CORE_API BaseSimulationExporter : public virtual BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseSimulationExporter, BaseObject);

    DataFileName       d_filename ;
    Data<unsigned int> d_exportEveryNbSteps;
    Data<bool>         d_exportAtBegin;
    Data<bool>         d_exportAtEnd;
    Data<bool>         d_isEnabled; ///< Enable or disable the component. (default=true)

    /// Don't override this function anymore. But you can do you init in the doInit.
    virtual void init() override final;

    /// Don't override this function anymore. But you can do your reinit in the doReInit.
    virtual void reinit() override final ;

    virtual void cleanup() override ;
    virtual void bwdInit() override ;
    virtual void handleEvent(Event *event) override ;

    virtual void doInit() {}
    virtual void doReInit() {}
    virtual bool write() = 0 ;


protected:
    BaseSimulationExporter() ;
    virtual ~BaseSimulationExporter() { }

    const std::string getOrCreateTargetPath(const std::string& filename, bool autonumbering) ;
    void updateFromDataField() ;
    unsigned int       m_stepCounter {0} ;
};

} /// namespace _baseexporter_

using _basesimulationexporter_::BaseSimulationExporter ;

} /// namespace simulation

} /// namespace sofa

#endif /// SOFA_CORE_EXPORTER_BASEEXPORTER_H
