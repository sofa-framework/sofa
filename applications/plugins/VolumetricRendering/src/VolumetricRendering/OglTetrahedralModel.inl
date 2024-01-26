/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef OGLTETRAHEDRALMODEL_INL_
#define OGLTETRAHEDRALMODEL_INL_

#include "OglTetrahedralModel.h"

#include <sstream>
#include <sofa/gl/GLSLShader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/BoundingBox.h>
#include <limits>

namespace sofa
{
namespace component
{
namespace visualmodel
{

template<class DataTypes>
OglTetrahedralModel<DataTypes>::OglTetrahedralModel()
    : m_topology(NULL)
    , m_positions(initData(&m_positions, "position", "Vertices coordinates"))
    , depthTest(initData(&depthTest, (bool)false, "depthTest", "Set Depth Test"))
    , blending(initData(&blending, (bool)false, "blending", "Set Blending"))
    , modified(false)
    , lastMeshRev(-1)
    , useTopology(false)
{
}

template<class DataTypes>
OglTetrahedralModel<DataTypes>::~OglTetrahedralModel()
{
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    m_topology = context->getMeshTopology();

    //instanciate the mapping tables
    //Useful for the PT algorithm only
    sofa::type::vector<sofa::gl::component::shader::OglFloatVector4Variable::SPtr > listVec4Variables;
    this->getContext()->core::objectmodel::BaseContext::template get<sofa::gl::component::shader::OglFloatVector4Variable, sofa::type::vector<sofa::gl::component::shader::OglFloatVector4Variable::SPtr> >
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
        msg_info() << "No MappingTable found, instanciating one";
        m_mappingTableValues = sofa::core::objectmodel::New<sofa::gl::component::shader::OglFloatVector4Variable>();
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
        msg_info() << "No RunSelectTable found, instanciating one";

        m_runSelectTableValues = sofa::core::objectmodel::New<sofa::gl::component::shader::OglFloatVector4Variable>();
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
        // currently OglTetrahedralMedal has to use topology to initialize data
        msg_error() << "No BaseMeshTopology found.";
        return;
    }
    // for now, no mesh file will be loaded directly from OglTetrahedralModel component
    // so force useTopology and modified to be true to enable the first time data loading from topology
    useTopology = true;
    modified = true;
    VisualModel::init();
    updateVisual();

}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::initVisual()
{
    const type::vector<Coord> tmpvertices = m_positions.getValue();
    type::vector<type::Vec3f> vertices;

    for (unsigned int i = 0; i<tmpvertices.size(); i++)
    {
        vertices.push_back(type::Vec3f(tmpvertices[i][0], tmpvertices[i][1], tmpvertices[i][2]));
    }

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

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::updateVisual()
{
    // Workaround if updateVisual() is called without an opengl context
    const auto* vparams = core::visual::VisualParams::defaultInstance();
    if (!vparams->isSupported(core::visual::API_OpenGL))
    {
        return;
    }

    if ((modified && !m_positions.getValue().empty())
        || useTopology)
    {
        // update mesh either when data comes from useTopology initially or vertices
        // get modified
        if (useTopology)
        {
            sofa::core::topology::TopologyModifier* topoMod;
            getContext()->get(topoMod);

            if (topoMod)
            {// topology will be handled by handleTopologyChange() with topologyModifier
                useTopology = false;
                computeMesh();
            }
            else if (topoMod == NULL&& m_topology->getRevision() != lastMeshRev)
            {
                computeMesh();
            }
        }
        modified = false;
    }
    m_tetrahedrons.updateIfDirty();
    m_positions.updateIfDirty();

    updateVertexBuffer();
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::computeMesh()
{
    using sofa::core::behavior::BaseMechanicalState;
    // update m_positions
    if (m_topology->hasPos())
    {
        msg_info() << "Copying " << m_topology->getNbPoints() << "points from topology.";
        helper::WriteAccessor<  Data<type::vector<Coord> > > position = m_positions;
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
        msg_info() << "Copying " << mstate->getSize() << " points from mechanical state.";
        helper::WriteAccessor< Data<type::vector<Coord> > > position = m_positions;
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
        msg_error() << "OglTetrahedralModel: can not update vertices!";
    }
    lastMeshRev = m_topology->getRevision();
    // update m_tetrahedrons
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& inputTetrahedrons = m_topology->getTetrahedra();
    if (this->f_printLog.getValue())
        msg_info() << "Copying " << inputTetrahedrons.size() << " tetrahedrons from topology.";
    helper::WriteAccessor< Data< type::vector<Tetrahedron> > > tetrahedrons = m_tetrahedrons;
    tetrahedrons.resize(inputTetrahedrons.size());
    for (unsigned int i = 0; i<inputTetrahedrons.size(); i++) {
        tetrahedrons[i] = inputTetrahedrons[i];
    }
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::drawTransparent(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;
    if (m_topology == NULL) return;
    if (m_topology->getNbTetrahedra() < 1) return;

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    if (vparams->displayFlags().getShowWireFrame())
    {
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        //glEnable(GL_CULL_FACE);
        //glCullFace(GL_FRONT);
    }

    if (blending.getValue())
        glEnable(GL_BLEND);

    if (depthTest.getValue())
    {
        glDepthFunc(GL_NEVER);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
    }

    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

#ifdef GL_LINES_ADJACENCY_EXT

    const core::topology::BaseMeshTopology::SeqTetrahedra& vec = m_topology->getTetrahedra();

    Coord v;

    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);

    int gltype = GL_FLOAT;

    if constexpr (std::is_same_v<typename Coord::value_type, double>)
    {
        gltype = GL_DOUBLE;
    }

    glVertexPointer(3, gltype, 0, nullptr);

    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    glEnableClientState(GL_VERTEX_ARRAY);

    glDrawElements(GL_LINES_ADJACENCY_EXT, m_topology->getNbTetrahedra() * 4, GL_UNSIGNED_INT, vec.data());


#else
    //
#endif

    if (vparams->displayFlags().getShowWireFrame())
    {
        //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        //glDisable(GL_CULL_FACE);
    }

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glPopAttrib();
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::computeBBox(const core::ExecParams * params, bool /* onlyVisible */)
{
    const type::vector<Coord>& position = m_positions.getValue();

    if (m_topology && position.size() > 0)
    {
        const SReal max_real = std::numeric_limits<SReal>::max();
        const SReal min_real = std::numeric_limits<SReal>::lowest();

        SReal maxBBox[3] = { min_real,min_real,min_real };
        SReal minBBox[3] = { max_real,max_real,max_real };

        for(const auto& tetra : m_topology->getTetrahedra())
        {
            for (unsigned int i = 0; i< 4; i++)
            {
                const auto& v = position[tetra[i]];

                if (minBBox[0] > v[0]) minBBox[0] = v[0];
                if (minBBox[1] > v[1]) minBBox[1] = v[1];
                if (minBBox[2] > v[2]) minBBox[2] = v[2];
                if (maxBBox[0] < v[0]) maxBBox[0] = v[0];
                if (maxBBox[1] < v[1]) maxBBox[1] = v[1];
                if (maxBBox[2] < v[2]) maxBBox[2] = v[2];
            }
        }

        this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox, maxBBox));
    }
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::updateVertexBuffer()
{
    const type::vector<Coord> tmpvertices = m_positions.getValue();
    type::vector<type::Vec3f> vertices;
    for (unsigned int i = 0; i<tmpvertices.size(); i++)
    {
        vertices.push_back(type::Vec3f(tmpvertices[i][0], tmpvertices[i][1], tmpvertices[i][2]));
    }

    unsigned positionsBufferSize;

    positionsBufferSize = (vertices.size()*sizeof(vertices[0]));

    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);
    //Positions
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        0,
        positionsBufferSize,
        vertices.data());

    glBindBufferARB(GL_ARRAY_BUFFER, 0);

}


} // namespace visualmodel

} // namesapce component

} // namespace sofa

#endif //OGLTETRAHEDRALMODEL_H_
