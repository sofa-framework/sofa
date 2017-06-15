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
#ifndef SOFA_COMPONENT_MISC_MESHEXPORTER_H
#define SOFA_COMPONENT_MISC_MESHEXPORTER_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/helper/OptionsGroup.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API MeshExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MeshExporter,core::objectmodel::BaseObject);

protected:
    sofa::core::topology::BaseMeshTopology* topology;
    sofa::core::behavior::BaseMechanicalState* mstate;
    unsigned int stepCounter;

    int nbFiles;

    std::string getMeshFilename(const char* ext);

public:
    sofa::core::objectmodel::DataFileName meshFilename;
    Data<sofa::helper::OptionsGroup> fileFormat;
    Data<defaulttype::Vec3Types::VecCoord> position;
    Data<bool> writeEdges;
    Data<bool> writeTriangles;
    Data<bool> writeQuads;
    Data<bool> writeTetras;
    Data<bool> writeHexas;
    //Data<helper::vector<std::string> > dPointsDataFields;
    //Data<helper::vector<std::string> > dCellsDataFields;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;

    helper::vector<std::string> pointsDataObject;
    helper::vector<std::string> pointsDataField;
    helper::vector<std::string> pointsDataName;

    helper::vector<std::string> cellsDataObject;
    helper::vector<std::string> cellsDataField;
    helper::vector<std::string> cellsDataName;
protected:
    MeshExporter();
    virtual ~MeshExporter();
public:
    void writeMesh();
    void writeMeshVTKXML();
    void writeMeshVTK();
    void writeMeshGmsh();
    void writeMeshNetgen();
    void writeMeshTetgen();

    void init();
    void cleanup();
    void bwdInit();

    void handleEvent(sofa::core::objectmodel::Event *);
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_MESHEXPORTER_H
