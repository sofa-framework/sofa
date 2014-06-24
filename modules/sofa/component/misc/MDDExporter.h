/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifndef MDDEXPORTER_H_
#define MDDEXPORTER_H_

#include <sofa/helper/helper.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/helper/list.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API MDDExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MDDExporter,core::objectmodel::BaseObject);

private:
    sofa::core::visual::VisualModel* vmodel;
    sofa::core::objectmodel::BaseContext* context;
    
    unsigned int stepCounter;
    unsigned int maxStep;
    unsigned int frameCount;
    unsigned int lastChangesCount;
    unsigned int nbVertex;
    helper::vector< unsigned int > pointI;
    helper::vector< helper::vector< unsigned int > > ancestorsI;
    vector< defaulttype::Vec3Types::VecCoord > vecFrame;
    
    std::ofstream* outfile;
    
    void writeMDD();
    void writeOBJ();
    void getState();
    
public:
    sofa::core::objectmodel::DataFileName mddFilename;
    Data< defaulttype::Vec3Types::VecCoord > m_position;
    Data< defaulttype::Vec3Types::VecCoord > m_tposition;
    Data< vector< core::topology::Triangle > > m_triangle;
    Data< vector< core::topology::Quad > > m_quad;
    Data< double > m_deltaT;
    Data< double > m_time;
    Data< std::string > m_name;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportOBJ;
    Data<bool> exportAtEnd;
    Data<bool> activateExport;
    
protected:
    MDDExporter();
    virtual ~MDDExporter();
public:
    
    void init();
    void cleanup();
    
    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* MDDEXPORTER_H_ */
