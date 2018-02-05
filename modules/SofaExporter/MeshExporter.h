/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_MISC_MESHEXPORTER_H
#define SOFA_COMPONENT_MISC_MESHEXPORTER_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/simulation/BaseSimulationExporter.h>

///////////////////////////// FORWARD DECLARATION //////////////////////////////////////////////////
namespace sofa {
    namespace core {
        namespace objectmodel {
            class Event ;
        }
        namespace behavior {
            class BaseMechanicalState;
        }
        namespace topology {
            class BaseMeshTopology ;
        }
    }
}



////////////////////////////////// DECLARATION /////////////////////////////////////////////////////
namespace sofa
{

namespace component
{

namespace _meshexporter_
{

using sofa::core::behavior::BaseMechanicalState ;
using sofa::core::objectmodel::Event ;
using sofa::core::topology::BaseMeshTopology ;
using sofa::simulation::BaseSimulationExporter ;

class SOFA_EXPORTER_API MeshExporter : public BaseSimulationExporter
{
public:
    SOFA_CLASS(MeshExporter, BaseSimulationExporter);

public:
    Data<sofa::helper::OptionsGroup> d_fileFormat;
    Data<defaulttype::Vec3Types::VecCoord> d_position;
    Data<bool> d_writeEdges;
    Data<bool> d_writeTriangles;
    Data<bool> d_writeQuads;
    Data<bool> d_writeTetras;
    Data<bool> d_writeHexas;

    helper::vector<std::string> pointsDataObject;
    helper::vector<std::string> pointsDataField;
    helper::vector<std::string> pointsDataName;

    helper::vector<std::string> cellsDataObject;
    helper::vector<std::string> cellsDataField;
    helper::vector<std::string> cellsDataName;

    virtual void doInit() override ;
    virtual void doReInit() override ;
    virtual void handleEvent(Event *) override ;

    virtual bool write() override ;

    bool writeMesh();
    bool writeMeshVTKXML();
    bool writeMeshVTK();
    bool writeMeshGmsh();
    bool writeMeshNetgen();
    bool writeMeshTetgen();


protected:
    MeshExporter();
    virtual ~MeshExporter();

    BaseMeshTopology*     m_inputtopology {nullptr};
    BaseMechanicalState*  m_inputmstate {nullptr};

    std::string getMeshFilename(const char* ext);
};

} /// namespace _meshexporter_

//todo(18.06): remove the old namespaces...
/// Import the object in the "old" namespace to allow smooth update of code base.
namespace misc {
    using _meshexporter_::MeshExporter ;
}

/// Import the object in the exporter namespace to avoid having all the object straight in component.
namespace exporter {
    using _meshexporter_::MeshExporter ;
}

} /// namespace component

} /// namespace sofa

#endif // SOFA_COMPONENT_MISC_MESHEXPORTER_H
