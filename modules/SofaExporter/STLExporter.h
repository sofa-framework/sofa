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
#ifndef STLEXPORTER_H_
#define STLEXPORTER_H_
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/core/objectmodel/GUIEvent.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API STLExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(STLExporter,core::objectmodel::BaseObject);

private:
    sofa::core::topology::BaseMeshTopology* topology;
    sofa::core::behavior::BaseMechanicalState* mstate;
    sofa::core::visual::VisualModel* vmodel;
    
    unsigned int stepCounter;
    unsigned int maxStep;

    std::ofstream* outfile;

    void writeSTL();
    void writeSTLBinary();
    
    int nbFiles;

public:
    sofa::core::objectmodel::DataFileName stlFilename;
    Data<bool> m_fileFormat;      //0 for Ascii Formats, 1 for Binary File Format
    Data<defaulttype::Vec3Types::VecCoord> m_position;
    Data< helper::vector< core::topology::BaseMeshTopology::Triangle > > m_triangle;
    Data< helper::vector< core::topology::BaseMeshTopology::Quad > > m_quad;
    
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;

protected:
    STLExporter();
    virtual ~STLExporter();
public:
    void init();
    void cleanup();
    void bwdInit();

    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* STLEXPORTER_H_ */
