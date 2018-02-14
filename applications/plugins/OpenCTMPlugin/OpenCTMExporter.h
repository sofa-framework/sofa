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

#ifndef OPENCTM_PLUGIN_OPENCTMEXPORTER_H
#define OPENCTM_PLUGIN_OPENCTMEXPORTER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <OpenCTMPlugin/config.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

/**
 * OpenCTMExporter class interfaces OpenCTM mesh writer with SOFA components.
 * For more information about the class API see doc: http://openctm.sourceforge.net/apidocs/
 *
 *  Created on: July 27th 2015
 *      Author: epernod
 */
class SOFA_OPENCTMPLUGIN_API OpenCTMExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(OpenCTMExporter, core::objectmodel::BaseObject);

    void init();
    void cleanup();
    void bwdInit();
    void handleEvent(sofa::core::objectmodel::Event *);

protected:
    /// Default constructor of the component
    OpenCTMExporter();

    /// Component destructor
    virtual ~OpenCTMExporter();

    /// Main internal method, implement the writing of CTM mesh file.
    void writeOpenCTM();


public:
    // Data filename where to export CTM file
    sofa::core::objectmodel::DataFileName m_outFilename;
    // Option to set the export of the CTM at the start of the program
    Data<bool> m_exportAtBegin; ///< export file at the initialization
    // Option to set the export of the CTM at the close of the program
    Data<bool> m_exportAtEnd; ///< export file when the simulation is finished
    // Option to use visual model instead of topology and mechanical components
    Data<bool> m_useVisualModel; ///< export file using information from current node visual model

private:
    // Current Node Topology component pointer (loaded at init)
    sofa::core::topology::BaseMeshTopology* m_pTopology;
    // Current Node Mechanical component pointer (loaded at init)
    sofa::core::behavior::BaseMechanicalState* m_pMstate;

    // Current Node visual model component pointer (loaded at init), if \sa m_useVisualModel is true
    sofa::component::visualmodel::VisualModelImpl* m_pVisual;

};

}

}

}

#endif /* OPENCTM_PLUGIN_OPENCTMEXPORTER_H */
