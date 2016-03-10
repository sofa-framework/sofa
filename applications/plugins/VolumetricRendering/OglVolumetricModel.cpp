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
#define OGLVOLUMETRICMODEL_CPP_

#include "OglVolumetricModel.h"

#include <sstream>
#include <limits>

#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{


SOFA_DECL_CLASS(OglVolumetricModel)

int OglVolumetricModelClass = sofa::core::RegisterObject("Volumetric model for OpenGL display")
.add < OglVolumetricModel >();



OglVolumetricModel::OglVolumetricModel()
    : b_modified(false)
    , m_lastMeshRev(-1)
    , b_useTopology(false)
    , depthTest(initData(&depthTest, (bool)false, "depthTest", "Set Depth Test"))
    , blending(initData(&blending, (bool)false, "blending", "Set Blending"))
{
}

OglVolumetricModel::~OglVolumetricModel()
{
}

void OglVolumetricModel::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    m_topology = context->getMeshTopology();

    //instanciate the mapping tables
    //Useful for the PT algorithm only
    sofa::helper::vector<sofa::component::visualmodel::OglFloatVector4Variable::SPtr > listVec4Variables;
    this->getContext()->core::objectmodel::BaseContext::template get<sofa::component::visualmodel::OglFloatVector4Variable, sofa::helper::vector<sofa::component::visualmodel::OglFloatVector4Variable::SPtr> >
        (&listVec4Variables, core::objectmodel::BaseContext::Local);
    for (unsigned int i = 0; i<listVec4Variables.size(); i++)
    {
        std::string idVec4 = listVec4Variables[i]->getId();

        if (!m_mappingTableValues)
        {
            if (idVec4.compare("MappingTable") == 0)
                m_mappingTableValues = listVec4Variables[i];
        }
        if (!m_runSelectTableValues)
        {
            if (idVec4.compare("RunSelectTable") == 0)
                m_runSelectTableValues = listVec4Variables[i];
        }
    }

    if (!m_mappingTableValues)
    {
        sout << "No MappingTable found, instanciating one" << sendl;
        m_mappingTableValues = sofa::core::objectmodel::New<sofa::component::visualmodel::OglFloatVector4Variable>();
        m_mappingTableValues->setName("MappingTable");
        m_mappingTableValues->setID("MappingTable");

        std::ostringstream oss;

        oss << "1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 "
            << "1 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 "
            << "1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 "
            << "1 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 "
            << "1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 "
            << "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 "
            << "0 0 1 0 1 0 0 0 0 0 0 1 0 1 0 0 "
            << "0 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0";
        m_mappingTableValues->value.read(oss.str());
        this->getContext()->addObject(m_mappingTableValues);
        m_mappingTableValues->init();


    }
    if (!m_runSelectTableValues)
    {
        sout << "No RunSelectTable found, instanciating one" << sendl;

        m_runSelectTableValues = sofa::core::objectmodel::New<sofa::component::visualmodel::OglFloatVector4Variable>();
        m_runSelectTableValues->setName("RunSelectTable");
        m_runSelectTableValues->setID("RunSelectTable");

        std::ostringstream oss;
        oss << "0. 1. 0. 0. "
            << "0. 0. 0. 0. "
            << "0. 0. 1. 0. "
            << "0. 0. 1. 0. "
            << "1. 0. 0. 0. "
            << "1. 0. 0. 0. "
            << "0. 0. 0. 1. "
            << "0. 0. 0. 1. "
            << "0. 0. 1. 0. "
            << "0. 1. 0. 0. ";
        m_runSelectTableValues->value.read(oss.str());
        this->getContext()->addObject(m_runSelectTableValues);
        m_runSelectTableValues->init();
    }

    if (!m_topology)
    {
        sout << "No BaseMeshTopology found." << sendl;
        b_useTopology = false;
        b_modified = false;
    }
    else
    {
        b_useTopology = true;
        b_modified = true;
    }

    VisualModel::init();
    updateVisual();

}

void OglVolumetricModel::initVisual()
{
    const defaulttype::ResizableExtVector<Coord>& vertices = m_positions.getValue();

    m_mappingTableValues->initVisual();
    m_runSelectTableValues->initVisual();

    glGenBuffersARB(1, &m_vbo);
    unsigned positionsBufferSize;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));
    unsigned int totalSize = positionsBufferSize;

    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);
    //Vertex Buffer creation
    glBufferDataARB(GL_ARRAY_BUFFER,
        totalSize,
        NULL,
        GL_DYNAMIC_DRAW);


    updateVertexBuffer();
    glBindBufferARB(GL_ARRAY_BUFFER, 0);
}

void OglVolumetricModel::updateVisual()
{
    if ((b_modified && !m_positions.getValue().empty())
        || b_useTopology)
    {
        // update mesh either when data comes from b_useTopology initially or vertices
        // get modified
        if(b_useTopology && m_topology->getRevision() != m_lastMeshRev)
            computeMesh();
        
        b_modified = false;
    }

    d_tetrahedra.updateIfDirty();
    d_hexahedra.updateIfDirty();

    m_positions.updateIfDirty();

    updateVertexBuffer();
}

void OglVolumetricModel::computeMesh()
{
    using sofa::core::behavior::BaseMechanicalState;

    if (b_useTopology)
    {
        // update m_positions
        if (m_topology->hasPos())
        {
            sout << "OglVolumetricModel: copying " << m_topology->getNbPoints() << "points from topology." << sendl;
            helper::WriteAccessor<  Data<defaulttype::ResizableExtVector<Coord> > > position = m_positions;
            position.resize(m_topology->getNbPoints());
            for (unsigned int i = 0; i<position.size(); i++) 
            {
                position[i][0] = (Real)m_topology->getPX(i);
                position[i][1] = (Real)m_topology->getPY(i);
                position[i][2] = (Real)m_topology->getPZ(i);
            }
        }
        else if (BaseMechanicalState* mstate = dynamic_cast< BaseMechanicalState* >(m_topology->getContext()->getMechanicalState()))
        {
            sout << "OglVolumetricModel: copying " << mstate->getSize() << " points from mechanical state." << sendl;
            helper::WriteAccessor< Data<defaulttype::ResizableExtVector<Coord> > > position = m_positions;
            position.resize(mstate->getSize());
            for (unsigned int i = 0; i<position.size(); i++)
            {
                position[i][0] = (Real)mstate->getPX(i);
                position[i][1] = (Real)mstate->getPY(i);
                position[i][2] = (Real)mstate->getPZ(i);
            }
        }
        else
        {
            serr << "OglVolumetricModel: can not update vertices!" << sendl;
        }

        m_lastMeshRev = m_topology->getRevision();

        // update Tetrahedrons
        const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& inputTetrahedra = m_topology->getTetrahedra();
        sout << "OglVolumetricModel: copying " << inputTetrahedra.size() << " tetrahedra from topology." << sendl;
        helper::WriteAccessor< Data< defaulttype::ResizableExtVector<Tetrahedron> > > tetrahedra = d_tetrahedra;
        tetrahedra.resize(inputTetrahedra.size());
        for (unsigned int i = 0; i<inputTetrahedra.size(); i++) {
            tetrahedra[i] = inputTetrahedra[i];
        }

        // update Hexahedrons
        const sofa::core::topology::BaseMeshTopology::SeqHexahedra& inputHexahedra = m_topology->getHexahedra();
        sout << "OglVolumetricModel: copying " << inputHexahedra.size() << " hexahedra from topology." << sendl;
        helper::WriteAccessor< Data< defaulttype::ResizableExtVector<Hexahedron> > > hexahedra = d_hexahedra;
        hexahedra.resize(hexahedra.size());
        for (unsigned int i = 0; i<inputHexahedra.size(); i++) {
            hexahedra[i] = inputHexahedra[i];
        }
    }

    //else we assumed all data are correctly set
}

void OglVolumetricModel::drawTransparent(const core::visual::VisualParams* vparams)
{
    using sofa::component::topology::TetrahedronSetTopologyContainer;
    if (!vparams->displayFlags().getShowVisualModels()) return;
    if (m_topology == NULL) return;
    if (m_topology->getNbTetrahedra() < 1) return;

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    if (vparams->displayFlags().getShowWireFrame())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
    }

    if (blending.getValue())
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    if (depthTest.getValue())
    {
        glDepthFunc(GL_NEVER);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
    }

#ifdef GL_LINES_ADJACENCY_EXT
    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);

    glVertexPointer(3, GL_FLOAT, 0, (char*)NULL + 0);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    glEnableClientState(GL_VERTEX_ARRAY);

    //helper::ReadAccessor< Data< defaulttype::ResizableExtVector<Tetrahedron> > > tetrahedra = d_tetrahedra;
    const defaulttype::ResizableExtVector<Tetrahedron>& tetrahedra = d_tetrahedra.getValue();

    glDrawElements(GL_LINES_ADJACENCY_EXT, tetrahedra.size() * 4, GL_UNSIGNED_INT, tetrahedra.begin());

#else
    //
#endif

    if (vparams->displayFlags().getShowWireFrame())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_CULL_FACE);
    }

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glPopAttrib();
}

void OglVolumetricModel::computeBBox(const core::ExecParams * params, bool /* onlyVisible */)
{
    if (m_topology)
    {
        Coord v;
        const defaulttype::ResizableExtVector<Coord>& position = m_positions.getValue();
        const SReal max_real = std::numeric_limits<SReal>::max();
        const SReal min_real = std::numeric_limits<SReal>::min();

        SReal maxBBox[3] = { min_real,min_real,min_real };
        SReal minBBox[3] = { max_real,max_real,max_real };

        for (unsigned int i = 0; i< position.size(); i++)
        {
            v = position[i];

            if (minBBox[0] > v[0]) minBBox[0] = v[0];
            if (minBBox[1] > v[1]) minBBox[1] = v[1];
            if (minBBox[2] > v[2]) minBBox[2] = v[2];
            if (maxBBox[0] < v[0]) maxBBox[0] = v[0];
            if (maxBBox[1] < v[1]) maxBBox[1] = v[1];
            if (maxBBox[2] < v[2]) maxBBox[2] = v[2];
        }
        this->f_bbox.setValue(params, sofa::defaulttype::TBoundingBox<SReal>(minBBox, maxBBox));
    }
}

void OglVolumetricModel::updateVertexBuffer()
{
    const defaulttype::ResizableExtVector<Coord>& vertices = m_positions.getValue();

    unsigned positionsBufferSize;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));

    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);
    //Positions
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        0,
        positionsBufferSize,
        vertices.getData());

    glBindBufferARB(GL_ARRAY_BUFFER, 0);

}

} // namespace visualmodel

} // namesapce component

} // namespace sofa

