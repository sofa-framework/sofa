/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef STEPEXPORTER_H_
#define STEPEXPORTER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <SofaBaseVisual/VisualModelImpl.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API STEPExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(STEPExporter,core::objectmodel::BaseObject);

private:
    sofa::core::topology::BaseMeshTopology* topology;
    sofa::core::behavior::BaseMechanicalState* mstate;
    sofa::core::visual::VisualModel* vmodel;
    
    unsigned int stepCounter;
    unsigned int maxStep;

    std::ofstream* outfile;

    void writeSTEP();
    void writeSTEPShort();
    
    int nbFiles;

public:
    sofa::core::objectmodel::DataFileName stepFilename;
    Data<bool> m_fileFormat;      //0 for Ascii Formats, 1 for Binary File Format
    Data< defaulttype::Vec3Types::VecCoord > m_position;
    Data< vector< core::topology::Triangle > > m_triangle;
    Data< vector< core::topology::Quad > > m_quad;
    
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;

protected:
    STEPExporter();
    virtual ~STEPExporter();
public:
    void init();
    void cleanup();
    void bwdInit();

    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* STEPEXPORTER_H_ */
