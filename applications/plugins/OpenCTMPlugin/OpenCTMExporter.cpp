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

#include <sofa/core/ObjectFactory.h>
#include "OpenCTMExporter.h"

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>

#include <openctm/openctm.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(OpenCTMExporter)

int OpenCTMExporterClass = core::RegisterObject("Export current topology under OpenCTM file format")
        .add< OpenCTMExporter >()
        ;

OpenCTMExporter::OpenCTMExporter()
    : m_outFilename( initData(&m_outFilename, "filename", "output CTM file name"))
    , m_exportAtBegin( initData(&m_exportAtBegin, false, "exportAtBegin", "export file at the initialization"))
    , m_exportAtEnd( initData(&m_exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished"))
    , m_useVisualModel ( initData(&m_useVisualModel, false, "useVisualModel", "export file using information from current node visual model"))
    , m_pTopology(NULL)
    , m_pMstate(NULL)
    , m_pVisual(NULL)
{
    this->f_listening.setValue(true);
}

OpenCTMExporter::~OpenCTMExporter()
{
    m_pTopology = NULL;
    m_pMstate = NULL;
    m_pVisual = NULL;
}

void OpenCTMExporter::init()
{
    // Get mechaOBj and topology components
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    if(m_useVisualModel.getValue())
    {
        context->get(m_pVisual);
        if (!m_pVisual)
        {
            serr << "OpenCTMExporter: Error: No visual model Object found." << sendl;
            return;
        }
        else
            sout << "OpenCTMExporter: Found visual model Object " << m_pVisual->getName() << sendl;
    }
    else
    {
        context->get(m_pMstate);
        if (!m_pMstate)
        {
            serr << "OpenCTMExporter: Error: No mechanical Object found." << sendl;
            return;
        }
        else
            sout << "OpenCTMExporter: Found mechanical Object " << m_pMstate->getName() << sendl;

        context->get(m_pTopology);
        if (!m_pTopology)
        {
                serr << "OpenCTMExporter: Error: No topology found." << sendl;
            return;
        }
        else
            sout << "OpenCTMExporter: found topology " << m_pTopology->getName() << sendl;
    }
}

void OpenCTMExporter::writeOpenCTM()
{
    std::string filename = m_outFilename.getFullPath();

    if(m_useVisualModel.getValue())
    {
        if (!m_pVisual)
            return;

        // Getting data access
        //typedef sofa::component::visualmodel::VisualModelImpl::TexCoord TexCoord;
        typedef sofa::component::visualmodel::VisualModelImpl::Coord Coord;

        const sofa::defaulttype::ResizableExtVector<Coord>& vertices = m_pVisual->getVertices();
        const sofa::defaulttype::ResizableExtVector<Coord>& normals = m_pVisual->getVnormals();
//        const sofa::defaulttype::ResizableExtVector<TexCoord>& texCoords = m_pVisual->getVtexcoords();

        const sofa::defaulttype::ResizableExtVector<sofa::core::topology::Triangle>& triangles = m_pVisual->getTriangles();
        const sofa::defaulttype::ResizableExtVector<sofa::core::topology::Quad>& quads = m_pVisual->getQuads();


        // Save the file using the OpenCTM API
        CTMexporter ctm;

        // creating CTM mesh memory
        CTMuint ctmVCount = vertices.size();

        // Filling ctm vertices buffer with current Data
        CTMfloat * ctmVertices = (CTMfloat *) malloc(3 * sizeof(CTMfloat) * ctmVCount);
        for (unsigned int i=0; i<ctmVCount; ++i)
        {
            ctmVertices[i*3] = vertices[i][0];
            ctmVertices[i*3 + 1] = vertices[i][1];
            ctmVertices[i*3 + 2] = vertices[i][2];
        }

        CTMuint ctmTriCount = 0;
        CTMuint * ctmTriangles = NULL;
        if (!triangles.empty()) // No triangles, trying quads
        {
            // Filling ctm triangles buffer with current Data
            ctmTriCount = triangles.size();
            ctmTriangles = (CTMuint *) malloc(3 * sizeof(CTMuint) * ctmTriCount);
            for (unsigned int i=0; i<ctmTriCount; ++i)
            {
                ctmTriangles[i*3] = triangles[i][0];
                ctmTriangles[i*3 + 1] = triangles[i][1];
                ctmTriangles[i*3 + 2] = triangles[i][2];
            }
        }
        else if (!quads.empty())
        {
            // Filling ctm triangles buffer with current Data
            ctmTriCount = quads.size()*2;
            ctmTriangles = (CTMuint *) malloc(3 * sizeof(CTMuint) * ctmTriCount);
            for (unsigned int i=0; i<quads.size(); ++i)
            {
                // tri-1
                ctmTriangles[i*6] = quads[i][0];
                ctmTriangles[i*6 + 1] = quads[i][1];
                ctmTriangles[i*6 + 2] = quads[i][2];

                // tri-2
                ctmTriangles[i*6 + 3] = quads[i][2];
                ctmTriangles[i*6 + 4] = quads[i][3];
                ctmTriangles[i*6 + 5] = quads[i][0];
            }
        }
        else
        {
            serr << "Error: no triangles nor quads found in component: " << m_pVisual->getName() << sendl;
            return;
        }


        // if has normals Filling ctm normals buffer with current Data
        CTMfloat * ctmNormals = NULL;
        if (!normals.empty())
        {
            ctmNormals = (CTMfloat *) malloc(3 * sizeof(CTMfloat) * ctmVCount);
            for (unsigned int i=0; i<ctmVCount; ++i)
            {
                ctmNormals[i*3] = normals[i][0];
                ctmNormals[i*3 + 1] = normals[i][1];
                ctmNormals[i*3 + 2] = normals[i][2];
            }
        }

        // if has texcoords Filling ctm normals buffer with current Data
//        if (!texCoords.empty())
//        {

//        }

        // creating CTM mesh memory
        std::cout << "ctmVCount: " << ctmVCount << std::endl;
        std::cout << "ctmTriCount: " << ctmTriCount << std::endl;


        ctm.DefineMesh(ctmVertices, ctmVCount, ctmTriangles, ctmTriCount, ctmNormals);
        ctm.Save(filename.c_str());
        free(ctmVertices);
        free(ctmTriangles);
    }
    else
    {
        if (!m_pMstate || !m_pTopology)
            return;

        // Save the file using the OpenCTM API
        CTMexporter ctm;

        // creating CTM mesh memory
        CTMuint ctmVCount = m_pMstate->getSize();
        CTMuint ctmTriCount = m_pTopology->getNbTriangles();

        CTMfloat * ctmVertices = (CTMfloat *) malloc(3 * sizeof(CTMfloat) * ctmVCount);
        CTMuint * ctmTriangles = (CTMuint *) malloc(3 * sizeof(CTMuint) * ctmTriCount);

        // Filling ctm buffer with current Data
        for (unsigned int i=0; i<ctmVCount; ++i)
        {
            ctmVertices[i*3] = m_pMstate->getPX(i);
            ctmVertices[i*3 + 1] = m_pMstate->getPY(i);
            ctmVertices[i*3 + 2] = m_pMstate->getPZ(i);
        }

        const sofa::core::topology::BaseMeshTopology::SeqTriangles& my_triangles = m_pTopology->getTriangles();
        for (unsigned int i=0; i<ctmTriCount; ++i)
        {
            ctmTriangles[i*3] = my_triangles[i][0];
            ctmTriangles[i*3 + 1] = my_triangles[i][1];
            ctmTriangles[i*3 + 2] = my_triangles[i][2];
        }

        ctm.DefineMesh(ctmVertices, ctmVCount, ctmTriangles, ctmTriCount, NULL);
        ctm.Save(filename.c_str());
        free(ctmVertices);
        free(ctmTriangles);
    }
}

void OpenCTMExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
        {
            std::cout <<"Writting Mesh" << std::endl;
            writeOpenCTM();
            break;
        }

        }
    }
}

void OpenCTMExporter::cleanup()
{
    if (m_exportAtEnd.getValue())
        writeOpenCTM();

}

void OpenCTMExporter::bwdInit()
{
    if (m_exportAtBegin.getValue())
        writeOpenCTM();
}

}

}

}
