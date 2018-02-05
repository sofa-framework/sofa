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

#include "STLExporter.h"

#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>

#include <sofa/core/ObjectFactory.h>


#include <sofa/core/behavior/BaseMechanicalState.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/GUIEvent.h>
#include <SofaBaseVisual/VisualModelImpl.h>

using sofa::core::objectmodel::KeypressedEvent ;
using sofa::simulation::AnimateBeginEvent ;
using sofa::simulation::AnimateEndEvent ;
using sofa::core::objectmodel::GUIEvent ;
using sofa::core::objectmodel::BaseContext ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::ComponentState ;

namespace sofa
{

namespace component
{

namespace _stlexporter_
{

SOFA_DECL_CLASS(STLExporter)

int STLExporterClass = core::RegisterObject("Save a topology in file")
        .add< STLExporter >();

STLExporter::STLExporter()
    : d_binaryFormat( initData(&d_binaryFormat, (bool)true, "binaryformat", "if true, save in binary format, otherwise in ascii"))
    , d_position( initData(&d_position, "position", "points coordinates"))
    , d_triangle( initData(&d_triangle, "triangle", "triangles indices"))
    , d_quad( initData(&d_quad, "quad", "quads indices"))
{
    this->addAlias(&d_triangle, "triangles");
    this->addAlias(&d_quad, "quads");
}

STLExporter::~STLExporter(){}

void STLExporter::doInit()
{
    doReInit() ;
}

void STLExporter::doReInit()
{
    BaseContext* context = this->getContext();

    context->get(m_inputtopology, BaseContext::Local);
    context->get(m_inputmstate, BaseContext::Local);
    context->get(m_inputvmodel, BaseContext::Local);

    // Test if the position has not been modified
    if(!d_position.isSet())
    {
        BaseData* pos = NULL;
        BaseData* tri = NULL;
        BaseData* qua = NULL;
        if(m_inputvmodel)
        {
            pos = m_inputvmodel->findData("position");
            tri = m_inputvmodel->findData("triangles");
            qua = m_inputvmodel->findData("quads");
        }
        else if(m_inputtopology)
        {
            pos = m_inputtopology->findData("position");
            tri = m_inputtopology->findData("triangles");
            qua = m_inputtopology->findData("quads");
        }
        else
        {
            msg_error() << "STLExporter needs VisualModel or Topology." ;
            m_componentstate = ComponentState::Invalid ;
            return ;
        }

        d_position.setParent(pos);
        d_triangle.setParent(tri);
        d_quad.setParent(qua);
    }
    m_componentstate = ComponentState::Valid ;
}

bool STLExporter::write()
{
    if(m_componentstate != ComponentState::Valid)
        return false ;

    if(d_binaryFormat.getValue())
        return writeSTL();
    return writeSTLBinary();
}

bool STLExporter::writeSTL(bool autonumbering)
{
    if(m_componentstate != ComponentState::Valid)
        return false ;

    std::string filename = getOrCreateTargetPath(d_filename.getValue(), d_exportEveryNbSteps.getValue() && autonumbering) ;
    filename += ".stl";

    std::ofstream outfile(filename.c_str());
    if( !outfile.is_open() )
    {
        msg_error() << "Unable to open file '" << filename << "'";
        return false;
    }

    helper::ReadAccessor< Data< helper::vector< BaseMeshTopology::Triangle > > > triangleIndices = d_triangle;
    helper::ReadAccessor< Data< helper::vector< BaseMeshTopology::Quad > > > quadIndices = d_quad;
    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > positionIndices = d_position;

    helper::vector< BaseMeshTopology::Triangle > vecTri;

    if(positionIndices.empty())
    {
        msg_error() << "No positions in topology." ;
        return false ;
    }
    if(!triangleIndices.empty())
    {
        for(unsigned int i=0;i<triangleIndices.size();i++)
        {
            vecTri.push_back(triangleIndices[i]);
        }
    }
    else if(!quadIndices.empty())
    {
        BaseMeshTopology::Triangle tri;
        for(unsigned int i=0;i<quadIndices.size();i++)
        {
            for(int j=0;j<3;j++)
            {
                tri[j] = quadIndices[i][j];
            }
            vecTri.push_back(tri);
            tri[0] = quadIndices[i][0];
            tri[1] = quadIndices[i][2];
            tri[2] = quadIndices[i][3];
            vecTri.push_back(tri);
        }
    }
    else
    {
        msg_error() << "No triangles nor quads in topology.";
        return false;
    }

    /* Get number of facets */
    const int nbt = vecTri.size();

    // Sets the floatfield format flag for the str stream to fixed
    std::cout.precision(6);

    /* solid */
    outfile << "solid Exported from Sofa" << std::endl;


    for(int i=0;i<nbt;i++)
    {
        /* normal */
        outfile << "facet normal 0 0 0" << std::endl;
        outfile << "outer loop" << std::endl;
        for (int j=0;j<3;j++)
        {
            /* vertices */
            outfile << "vertex " << std::fixed << positionIndices[ vecTri[i][j] ] << std::endl;
        }
        outfile << "endloop" << std::endl;
        outfile << "endfacet" << std::endl;
    }

    /* endsolid */
    outfile << "endsolid Exported from Sofa" << std::endl;

    outfile.close();

    msg_info() << "File '" << filename << "' written" ;
    return true ;
}

bool STLExporter::writeSTLBinary(bool autonumbering)
{
    if(m_componentstate != ComponentState::Valid)
        return false ;

    std::string filename = getOrCreateTargetPath(d_filename.getValue(),
                                                 d_exportEveryNbSteps.getValue() && autonumbering) ;
    filename += ".stl";

    std::ofstream outfile(filename.c_str(), std::ios::out | std::ios::binary);
    if( !outfile.is_open() )
    {
        msg_error() << "Unable to open file '" << filename << "'";
        return false;
    }

    helper::ReadAccessor< Data< helper::vector< BaseMeshTopology::Triangle > > > triangleIndices = d_triangle;
    helper::ReadAccessor< Data< helper::vector< BaseMeshTopology::Quad > > > quadIndices = d_quad;
    helper::ReadAccessor< Data< defaulttype::Vec3Types::VecCoord> > positionIndices = d_position;

    helper::vector< BaseMeshTopology::Triangle > vecTri;

    if(positionIndices.empty())
    {
        msg_error() << "No positions in topology." ;
        return false;
    }
    if(!triangleIndices.empty())
    {
        for(unsigned int i=0;i<triangleIndices.size();i++)
        {
            vecTri.push_back(triangleIndices[i]);
        }
    }
    else if(!quadIndices.empty())
    {
        BaseMeshTopology::Triangle tri;
        for(unsigned int i=0;i<quadIndices.size();i++)
        {
            for(int j=0;j<3;j++)
            {
                tri[j] = quadIndices[i][j];
            }
            vecTri.push_back(tri);
            tri[0] = quadIndices[i][0];
            tri[1] = quadIndices[i][2];
            tri[2] = quadIndices[i][3];
            vecTri.push_back(tri);
        }
    }
    else
    {
        msg_error() << "No triangles nor quads in topology.";
        return false;
    }

    // Sets the floatfield format flag for the str stream to fixed
    std::cout.precision(6);

    /* Creating header file */
    char* buffer = new char [80];
    // Cleaning buffer
    for(int i=0;i<80;i++)
    {
        buffer[i]='\0';
    }
    strcpy(buffer, "Exported from Sofa");
    outfile.write(buffer,80);

    /* Number of facets */
    const unsigned int nbt = vecTri.size();
    outfile.write((char*)&nbt,4);

    // Parsing facets
    for(unsigned long i=0;i<nbt;i++)
    {
        /* normal */
        float nul = 0.; // normals are set to 0
        outfile.write((char*)&nul, 4);
        outfile.write((char*)&nul, 4);
        outfile.write((char*)&nul, 4);
        for (int j=0;j<3;j++)
        {
            /* vertices */
            float iOne = (float)positionIndices[ vecTri[i][j] ][0];
            float iTwo = (float)positionIndices[ vecTri[i][j] ][1];
            float iThree = (float)positionIndices[ vecTri[i][j] ][2];
            outfile.write( (char*)&iOne, 4);
            outfile.write( (char*)&iTwo, 4);
            outfile.write( (char*)&iThree, 4);
        }

        /* Attribute byte count */
        // attribute count is currently not used, it's garbage
        unsigned int zero = 0;
        outfile.write((char*)&zero, 2);
    }

    outfile.close();
    msg_info() << "File '" << filename << "' written" ;
    return true;
}

void STLExporter::handleEvent(Event *event)
{
    if(m_componentstate != ComponentState::Valid)
        return ;

    if (KeypressedEvent::checkEventType(event))
    {
        KeypressedEvent *ev = static_cast<KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            //todo(18.06) remove the behavior
            msg_deprecated() << "Hard coded interaction behavior in component is now a deprecated behavior."
                                "Scene specific interaction should be implement using an external controller or pythonScriptController."
                                "Please update your scene because this behavior will be removed in Sofa 18.06";
            if(d_binaryFormat.getValue())
                writeSTLBinary(false);
            else
                writeSTL(false);
            break;
        }
    }


    if (GUIEvent::checkEventType(event))
    {
        GUIEvent *guiEvent = static_cast<GUIEvent *>(event);

        if (guiEvent->getValueName().compare("STLExport") == 0)
        {
            if(d_binaryFormat.getValue())
                writeSTLBinary(false);
            else
                writeSTL(false);
        }
    }

    BaseSimulationExporter::handleEvent(event) ;
}

} // namespace misc

} // namespace component

} // namespace sofa

