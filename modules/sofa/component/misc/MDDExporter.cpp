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

#include "MDDExporter.h"

#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>


namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(MDDExporter)

int MDDExporterClass = core::RegisterObject("Read State vectors from file")
        .add< MDDExporter >();

MDDExporter::MDDExporter()
    : stepCounter(0)
    , mddFilename( initData(&mddFilename, "filename", "output MDD file name"))
    , m_position( initData(&m_position, "position", "positions"))
    , m_deltaT( initData(&m_deltaT, "dt", "time step of the simulation"))
    , m_time( initData(&m_time, "time", "current time"))
    , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps"))
    , exportOBJ( initData(&exportOBJ, (bool)false, "exportOBJ", "optional, if true build an OBJ file from the Topology"))
    , activateExport( initData(&activateExport, (bool)false, "activateExport", "you can use the key 'e' to begin or stop the recording"))
{
    this->f_listening.setValue(true);
}

MDDExporter::~MDDExporter()
{
}

void MDDExporter::init()
{
    context = this->getContext();
    context->get(vmodel, sofa::core::objectmodel::BaseContext::SearchDown); // Visual Model
    
    // Activate the listening to the event in order to be able to export file at the nth-step
    this->f_listening.setValue(true);
    
    sofa::core::objectmodel::BaseData* pos = NULL;
    sofa::core::objectmodel::BaseData* delta = NULL;
    sofa::core::objectmodel::BaseData* time = NULL;
    sofa::core::objectmodel::BaseData* name = NULL;
    sofa::core::objectmodel::BaseData* tri = NULL;
    sofa::core::objectmodel::BaseData* qua = NULL;

    delta = context->findField("dt");
    time = context->findField("time");
    name = context->findField("name");
    
    if(!pos && vmodel)
    {
        pos = vmodel->findField("position");
        tri = vmodel->findField("triangles");
        qua = vmodel->findField("quads");
    }
    else
    {
        serr << "MDDExporter : Error, no VisualModel" << sendl;
        return;
    }
    if(!pos)
    {
        serr << "MDDExporter : Error, no positions in VisualModel" << sendl;
        return;
    }
    
    m_position.setParent(pos);
    m_deltaT.setParent(delta);
    m_time.setParent(time);
    m_name.setParent(name);
    m_triangle.setParent(tri);
    m_quad.setParent(qua);
    
    nbVertex = 0;
    frameCount = 0;
    lastChangesCount = 0;
    
}

void MDDExporter::getState()
{
    helper::ReadAccessor<Data< defaulttype::Vec3Types::VecCoord > > pointsIndices = m_position; // The current frame
    vmodel->updatePointAncestors(&pointI, &ancestorsI);
    if(activateExport.getValue())
    {
        if(vecFrame.size() != 0)
        {
            vecFrame.push_back(pointsIndices.ref());
            unsigned int nbChanges = pointI.size()-lastChangesCount;
            if( nbChanges > 0)
            {
                unsigned int firstFrameSize = vecFrame[0].size();
                sout << "A change occured : " << sendl;
                sout << "    " << vecFrame.size() << " frames to change" << sendl;
                sout << "    " << nbChanges << " changes to perform on each frame" << sendl;
                for(unsigned int i=0;i<vecFrame.size();i++)
                {
                    vecFrame[i].resize(firstFrameSize+nbChanges);
                    for(unsigned int j=nbChanges;j>0;--j)
                    {
                        double posX = 0;
                        double posY = 0;
                        double posZ = 0;
                        for(unsigned int k=0;k<ancestorsI[pointI.size()-j].size();k++)
                        {
                            posX += vecFrame[i][ancestorsI[pointI.size()-j][k]][0]/ancestorsI[pointI.size()-j].size();
                            posY += vecFrame[i][ancestorsI[pointI.size()-j][k]][1]/ancestorsI[pointI.size()-j].size();
                            posZ += vecFrame[i][ancestorsI[pointI.size()-j][k]][2]/ancestorsI[pointI.size()-j].size();
                        }
                        defaulttype::Vec3Types::Coord coordNewPoint;
                        coordNewPoint[0]=posX;
                        coordNewPoint[1]=posY;
                        coordNewPoint[2]=posZ;
                        vecFrame[i][vecFrame[i].size()-j] = coordNewPoint;
                    }
                }
                lastChangesCount = pointI.size();
            }
        }
        else
        {
            // First frame of the record
            vecFrame.push_back(pointsIndices.ref());
            sout << "Start recording mdd..." << sendl;
        }
        if(nbVertex!=vecFrame[vecFrame.size()-1].size())
        {
            nbVertex = vecFrame[vecFrame.size()-1].size();
            sout << nbVertex << " vertices" << sendl;
        }
        ++frameCount;
    }
    else if(vecFrame.size() == 0)
    {
        // If the record isn't started yet then we do not take into account the last topological changes
        lastChangesCount = pointI.size();
    }
}

void MDDExporter::writeMDD()
{
    std::string filename = mddFilename.getFullPath();
    filename += ".mdd";
    
    outfile = new std::ofstream(filename.c_str(), std::ios::out | std::ios::binary);
    if( !outfile->is_open() )
    {
        serr << "Error creating file " << filename << sendl;
        delete outfile;
        outfile = NULL;
        return;
    }
    
    helper::ReadAccessor<Data< defaulttype::Vec3Types::VecCoord > > pointsIndices = m_position;
    helper::ReadAccessor<Data< double > > deltaIndice = m_deltaT;
    helper::ReadAccessor<Data< double > > timeIndice = m_time;
    helper::ReadAccessor<Data< unsigned int> > nbStepIndice = exportEveryNbSteps;
    
    if(vecFrame.empty())
    {
        serr << "Error generating file " << filename << sendl;
        return;
    }
    
    /* Header */
    // Number of frame (int)
    unsigned int nbf = frameCount;
    char *cnbf = (char*)&nbf;
    outfile->write(&cnbf[3],1);
    outfile->write(&cnbf[2],1);
    outfile->write(&cnbf[1],1);
    outfile->write(&cnbf[0],1);
    
    // Number of vertex (int)
    unsigned int nbv = pointsIndices.size();
    char *cnbv = (char*)&nbv;
    outfile->write(&cnbv[3],1);
    outfile->write(&cnbv[2],1);
    outfile->write(&cnbv[1],1);
    outfile->write(&cnbv[0],1);
    
    /* Time step */
    // Time step for each frame in seconds(float)
    float dt = deltaIndice * nbStepIndice;
    char *cdt = (char*)&dt;
    for(unsigned int i=0;i<nbf;i++)
    {
        outfile->write(&cdt[3],1);
        outfile->write(&cdt[2],1);
        outfile->write(&cdt[1],1);
        outfile->write(&cdt[0],1);
    }
    
    /* Vertices */
    // Initial vertices
    for(unsigned int i=0;i<nbv;i++)
    {
        for(int j=0;j<3;j++)
        {
            // Vertex (float)
            float vertex = vecFrame[0][i][j];
            char *cvertex = (char*)&vertex;
            outfile->write(&cvertex[3],1);
            outfile->write(&cvertex[2],1);
            outfile->write(&cvertex[1],1);
            outfile->write(&cvertex[0],1);
        }
    }
    
    // vertices for each frame
    for(unsigned int i=0;i<nbf;i++)
    {
        if( i>0 && (vecFrame[i].size()!=vecFrame[i-1].size()) )
        {
            serr << "MDDExporter : Error, n-1=" << vecFrame[i-1].size() << "   n=" << vecFrame[i].size() << "   n+1=" << vecFrame[i+1].size() << sendl;
            serr << "MDDExporter : Error, mismatch in number of vertex in frame " << i << sendl;
            return;
        }
        for(unsigned int j=0;j<nbv;j++)
        {
            for(int k=0;k<3;k++)
            {
                // Vertex (float)
                float vertex = vecFrame[i][j][k];
                char *cvertex = (char*)&vertex;
                outfile->write(&cvertex[3],1);
                outfile->write(&cvertex[2],1);
                outfile->write(&cvertex[1],1);
                outfile->write(&cvertex[0],1);
            }
        }
    }
    
    outfile->close();
    std::cout << "number of frames : " << nbf << "      number of vertices : " << nbv << std::endl;
    std::cout << filename << " written" << std::endl;
}

void MDDExporter::writeOBJ()
{
    std::string filename = mddFilename.getFullPath();
    filename += ".obj";
    
    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file " << filename << sendl;
        delete outfile;
        outfile = NULL;
        return;
    }
    
    helper::ReadAccessor<Data< std::string > > nameIndice = m_name;
    helper::ReadAccessor<Data< vector< core::topology::Triangle > > > triangleIndices = m_triangle;
    helper::ReadAccessor<Data< vector< core::topology::Quad > > > quadIndices = m_quad;
    
    if(vecFrame.empty())
    {
        serr << "Error generating file " << filename << sendl;
        return;
    }
    
    *outfile << "g "<<nameIndice<<"\n";
    
    for (unsigned int i=0; i<vecFrame[0].size(); i++)
    {
        *outfile << "v "<< std::fixed << vecFrame[0][i][0]<<' '<< std::fixed <<vecFrame[0][i][1]<<' '<< std::fixed <<vecFrame[0][i][2]<<'\n';
    }
    
    for (unsigned int i = 0; i < triangleIndices.size() ; i++)
    {
        *outfile << "f";
        for (int j=0; j<3; j++)
        {
            *outfile << ' ' << triangleIndices[i][j]+1;
        }
        *outfile << '\n';
    }
    for (unsigned int i = 0; i < quadIndices.size() ; i++)
    {
        *outfile << "f";
        for (int j=0; j<4; j++)
        {
            *outfile << ' ' << quadIndices[i][j]+1;
        }
        *outfile << '\n';
    }
    *outfile << std::endl;
    outfile->close();
    std::cout << filename << " written" << std::endl;
}

void MDDExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
        maxStep = exportEveryNbSteps.getValue();
        if (maxStep == 0) return;
        
        stepCounter++;
        if( stepCounter % maxStep == 0 )
        {
            getState();
        }
    }
    if(core::objectmodel::KeypressedEvent* ev = dynamic_cast<core::objectmodel::KeypressedEvent*>(event))
    {
        switch(ev->getKey())
        {
            // The key E is pressed
            case 'E':
            case 'e':
            {
                activateExport.setValue(!activateExport.getValue());
                if(activateExport.getValue())
                {
                    std::cout << "begin record" << std::endl;
                }
                else
                {
                    std::cout << "stop record" << std::endl;
                }
            }
        }
    }
}

void MDDExporter::cleanup()
{
    if(!vecFrame.empty())
    {
        writeMDD();
        if(exportOBJ.getValue() != 0)
            writeOBJ();
    }
}


} // namespace misc

} // namespace component

} // namespace sofa
