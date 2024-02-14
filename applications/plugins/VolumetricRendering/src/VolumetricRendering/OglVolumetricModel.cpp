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
#define OGLVOLUMETRICMODEL_CPP_

#include <VolumetricRendering/OglVolumetricModel.h>

#include <sstream>
#include <limits>

#include <sofa/core/ObjectFactory.h>
#include <sofa/gl/GLSLShader.h>
#include <sofa/type/BoundingBox.h>
#include <sofa/gl/component/shader/OglAttribute.inl>

namespace sofa::component::visualmodel
{

int OglVolumetricModelClass = sofa::core::RegisterObject("Volumetric model for OpenGL display")
.add < OglVolumetricModel >();



OglVolumetricModel::OglVolumetricModel()
    : d_tetrahedra(initData(&d_tetrahedra, "tetrahedra", "Tetrahedra to draw"))
    , d_hexahedra(initData(&d_hexahedra, "hexahedra", "Hexahedra to draw"))
    , d_volumeScale(initData(&d_volumeScale, (float)1.0, "volumeScale", "Scale for each volumetric primitive"))
    , d_depthTest(initData(&d_depthTest, (bool)false, "depthTest", "Set Depth Test"))
    , d_blending(initData(&d_blending, (bool)false, "blending", "Set Blending"))
    , d_defaultColor(initData(&d_defaultColor, type::Vec4f(), "defaultColor", "Color for each volume (if the attribute a_vertexColor is not detected)"))
    , b_modified(false)
    , b_useTopology(false)
    , b_tboCreated(false)
{
    addAlias(&d_defaultColor, "color");
}

OglVolumetricModel::~OglVolumetricModel()
{
}

void OglVolumetricModel::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    m_topology = context->getMeshTopology();

    context->get(m_shader);
    if (!m_shader)
    {
        msg_error() << "Need an OglShader to work !";
        return;
    }

  

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
        context->addObject(m_mappingTableValues);
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
        context->addObject(m_runSelectTableValues);
        m_runSelectTableValues->init();
    }


    if (!m_topology)
    {
        msg_info() << "No BaseMeshTopology found.";
        b_useTopology = false;
        b_modified = false;
    }
    else
    {
        b_useTopology = true;
        b_modified = true;
    }

    VisualModel::init();

    if (b_useTopology)
        computeMeshFromTopology();


}

void OglVolumetricModel::initVisual()
{
    const type::vector<Coord>& positions = m_positions.getValue();

    m_mappingTableValues->initVisual();
    m_runSelectTableValues->initVisual();

    glGenBuffersARB(1, &m_vbo);
    unsigned positionsBufferSize;

    positionsBufferSize = (positions.size()*sizeof(positions[0]));
    unsigned int totalSize = positionsBufferSize;

    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);
    //Vertex Buffer creation
    glBufferDataARB(GL_ARRAY_BUFFER,
        totalSize,
        NULL,
        GL_DYNAMIC_DRAW);


    updateVertexBuffer();
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    //Check attributes
    sofa::type::vector<sofa::gl::component::shader::OglFloat4Attribute::SPtr > listVec4Attributes;
    this->getContext()->core::objectmodel::BaseContext::template get<sofa::gl::component::shader::OglFloat4Attribute, sofa::type::vector<sofa::gl::component::shader::OglFloat4Attribute::SPtr> >
        (&listVec4Attributes, core::objectmodel::BaseContext::Local);
    for (unsigned int i = 0; i < listVec4Attributes.size(); i++)
    {
        std::string idVec4 = listVec4Attributes[i]->getId();

        if (!m_vertexColors)
        {
            if (idVec4.compare("a_vertexColor") == 0)
                m_vertexColors = listVec4Attributes[i];
        }
    }
    if (!m_vertexColors)
    {
        msg_error() << "No attributes called a_vertexColor found, instanciating one with a default color";
        m_vertexColors = sofa::core::objectmodel::New<sofa::gl::component::shader::OglFloat4Attribute>();
        m_vertexColors->setName("a_vertexColor");
        m_vertexColors->setID("a_vertexColor");
        m_vertexColors->setIndexShader(0);
    }
    sofa::type::vector<type::Vec4f>& vertexColors = *(m_vertexColors->beginEdit());
    unsigned int nbPositions = m_positions.getValue().size();
    if ((vertexColors.size() != nbPositions))
    {
        const type::Vec4f& defaultColor = d_defaultColor.getValue();
        vertexColors.clear();
        for (unsigned int i = 0; i < nbPositions; i++)
        {
            vertexColors.push_back(defaultColor);
        }
    }

    getContext()->addObject(m_vertexColors);
    m_vertexColors->init();
    m_vertexColors->initVisual();

}

void OglVolumetricModel::updateVisual()
{
    // Workaround if updateVisual() is called without an opengl context
    const auto* vparams = core::visual::VisualParams::defaultInstance();
    if (!vparams->isSupported(core::visual::API_OpenGL))
    {
        return;
    }

    if (b_modified || d_tetrahedra.isDirty() || d_hexahedra.isDirty() || m_positions.isDirty())
    {
        //if(b_useTopology)
        //    computeMeshFromTopology();
        //else
        {
            d_tetrahedra.updateIfDirty();
            d_hexahedra.updateIfDirty();
            m_positions.updateIfDirty();
        }

        splitHexahedra();
        b_modified = false;
    }


    computeBarycenters();
    updateVertexBuffer();
    
}

void OglVolumetricModel::computeMeshFromTopology()
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
    else
        
    if (BaseMechanicalState* mstate = dynamic_cast< BaseMechanicalState* >(m_topology->getContext()->getMechanicalState()))
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
        msg_error() << "OglVolumetricModel: can not update positions!";
    }

    // update Tetrahedrons
    const SeqTetrahedra& inputTetrahedra = m_topology->getTetrahedra();
    msg_info() << "Copying " << inputTetrahedra.size() << " tetrahedra from topology.";
    helper::WriteAccessor< Data< type::vector<Tetrahedron> > > tetrahedra = d_tetrahedra;
    tetrahedra.clear();
    tetrahedra.resize(inputTetrahedra.size());
    for (unsigned int i = 0; i<inputTetrahedra.size(); i++) {
        tetrahedra[i] = inputTetrahedra[i];
    }
        
    // update Hexahedrons
    const SeqHexahedra& inputHexahedra = m_topology->getHexahedra();
    msg_info() << "Copying " << inputHexahedra.size() << " hexahedra from topology.";
    helper::WriteAccessor< Data< type::vector<Hexahedron> > > hexahedra = d_hexahedra;
    hexahedra.clear();
    hexahedra.resize(inputHexahedra.size());
    for (unsigned int i = 0; i < inputHexahedra.size(); i++) {
        hexahedra[i] = inputHexahedra[i];
    }
}

void OglVolumetricModel::splitHexahedra()
{
    helper::ReadAccessor< Data< type::vector<Tetrahedron> > > tetrahedra = d_tetrahedra;
    helper::ReadAccessor< Data< type::vector<Hexahedron> > > hexahedra = d_hexahedra;
    m_hexaToTetrahedra.clear();

    //split hexahedron to 5 tetrahedra
    for (unsigned int i = 0; i<hexahedra.size(); i++)
    {
        const Hexahedron& hex = hexahedra[i];
        Tetrahedron tet[5];
        tet[0][0] = hex[6]; tet[0][1] = hex[4]; tet[0][2] = hex[1]; tet[0][3] = hex[5];
        tet[1][0] = hex[4]; tet[1][1] = hex[3]; tet[1][2] = hex[1]; tet[1][3] = hex[0];
        tet[2][0] = hex[4]; tet[2][1] = hex[7]; tet[2][2] = hex[6]; tet[2][3] = hex[3];
        tet[3][0] = hex[3]; tet[3][1] = hex[6]; tet[3][2] = hex[1]; tet[3][3] = hex[2];
        tet[4][0] = hex[4]; tet[4][1] = hex[6]; tet[4][2] = hex[1]; tet[4][3] = hex[3];
        for (unsigned int j = 0; j < 5; j++)
            m_hexaToTetrahedra.push_back(tet[j]);
    }
}


void OglVolumetricModel::computeBarycenters()
{
    helper::ReadAccessor< Data< type::vector<Tetrahedron> > > tetrahedra = d_tetrahedra;
    helper::ReadAccessor< Data< type::vector<Hexahedron> > > hexahedra = d_hexahedra;
    const type::vector<Coord>& positions = m_positions.getValue();
    
    m_tetraBarycenters.clear();
    for (unsigned int i = 0; i < tetrahedra.size(); i++)
    {
        const Tetrahedron& t = tetrahedra[i];
        Coord barycenter = (positions[t[0]] + positions[t[1]] + positions[t[2]] + positions[t[3]])*0.25;
        m_tetraBarycenters.push_back(barycenter);
    }

    m_hexaBarycenters.clear();
    for (unsigned int i = 0; i < hexahedra.size(); i++)
    {
        const Hexahedron& h = hexahedra[i];
        Coord barycenter = (positions[h[0]] + positions[h[1]] + positions[h[2]] + positions[h[3]] +
            positions[h[4]] + positions[h[5]] + positions[h[6]] + positions[h[7]])*0.125;
        m_hexaBarycenters.push_back(barycenter);
        m_hexaBarycenters.push_back(barycenter);
        m_hexaBarycenters.push_back(barycenter);
        m_hexaBarycenters.push_back(barycenter);
        m_hexaBarycenters.push_back(barycenter);
    }
    unsigned int tetraBarycentersBufferSize = m_tetraBarycenters.size() * 3 * sizeof(GLfloat);
    unsigned int hexaBarycentersBufferSize = m_hexaBarycenters.size() * 3 * sizeof(GLfloat);
    if (!b_tboCreated)
    {
        //Texture buffer objects
        if (m_tetraBarycenters.size() > 0)
        {
            glGenBuffers(1, &m_tetraBarycentersTbo);
            glBindBuffer(GL_TEXTURE_BUFFER, m_tetraBarycentersTbo);
            glBufferData(GL_TEXTURE_BUFFER, tetraBarycentersBufferSize, &(m_tetraBarycenters[0]), GL_DYNAMIC_COPY);
            glGenTextures(1, &m_tetraBarycentersTboTexture);
            glBindBuffer(GL_TEXTURE_BUFFER, 0);
            b_tboCreated = true;
        }
        if (m_hexaBarycenters.size() > 0)
        {
            glGenBuffers(1, &m_hexaBarycentersTbo);
            glBindBuffer(GL_TEXTURE_BUFFER, m_hexaBarycentersTbo);
            glBufferData(GL_TEXTURE_BUFFER, hexaBarycentersBufferSize, &(m_hexaBarycenters[0]), GL_DYNAMIC_COPY);
            glGenTextures(1, &m_hexaBarycentersTboTexture);
            glBindBuffer(GL_TEXTURE_BUFFER, 0);
            b_tboCreated = true;
        }
    }

    if (m_tetraBarycenters.size() > 0)
    {
        glBindBuffer(GL_TEXTURE_BUFFER, m_tetraBarycentersTbo);
        glBufferSubData(GL_TEXTURE_BUFFER, 0, tetraBarycentersBufferSize, &(m_tetraBarycenters[0]));
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }
    if (m_hexaBarycenters.size() > 0)
    {
        glBindBuffer(GL_TEXTURE_BUFFER, m_hexaBarycentersTbo);
        glBufferSubData(GL_TEXTURE_BUFFER, 0, hexaBarycentersBufferSize, &(m_hexaBarycenters[0]));
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }
}

void OglVolumetricModel::handleTopologyChange()
{
    std::list< const core::topology::TopologyChange * >::const_iterator itBegin = m_topology->beginChange();
    std::list< const core::topology::TopologyChange * >::const_iterator itEnd = m_topology->endChange();

    //TODO one day: update only necessary things
    //but for now, if there is only one change, we update everything
    bool oneTime = true;
    for (std::list< const core::topology::TopologyChange * >::const_iterator changeIt = itBegin; changeIt != itEnd && oneTime; ++changeIt)
    {
        //const  core::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        b_modified = true;
        oneTime = false;
    }
}

void OglVolumetricModel::drawTransparent(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    if (vparams->displayFlags().getShowWireFrame())
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
    }

    if (d_blending.getValue())
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    if (d_depthTest.getValue())
    {
        glDepthFunc(GL_NEVER);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
    }

#ifdef GL_LINES_ADJACENCY_EXT
    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);

    int gltype = GL_FLOAT;

    if constexpr (std::is_same_v<typename Coord::value_type, double>)
    {
        gltype = GL_DOUBLE;
    }

    glVertexPointer(3, gltype, 0, nullptr);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);
    
    glEnableClientState(GL_VERTEX_ARRAY);

    const type::vector<Tetrahedron>& tetrahedra = d_tetrahedra.getValue();
    const type::vector<Hexahedron>& hexahedra = d_hexahedra.getValue();
    //glEnable(GL_CLIP_DISTANCE0);


    if (tetrahedra.size() > 0)
    {
        glBindTexture(GL_TEXTURE_BUFFER, m_tetraBarycentersTboTexture);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, m_tetraBarycentersTbo);
        glDrawElements(GL_LINES_ADJACENCY_EXT, tetrahedra.size() * 4, GL_UNSIGNED_INT, tetrahedra.data());
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }
    if(m_hexaToTetrahedra.size() > 0)
    {
        glBindTexture(GL_TEXTURE_BUFFER, m_hexaBarycentersTboTexture);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, m_hexaBarycentersTbo);
        glDrawElements(GL_LINES_ADJACENCY_EXT, m_hexaToTetrahedra.size() * 4, GL_UNSIGNED_INT, m_hexaToTetrahedra.data());
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }
    //glDisable(GL_CLIP_DISTANCE0);

#else
    msg_error() << "Your OpenGL driver does not support GL_LINES_ADJACENCY_EXT";
#endif // GL_LINES_ADJACENCY_EXT

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
    //if (m_topology)
    {
        Coord v;
        const type::vector<Coord>& position = m_positions.getValue();
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
        this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox, maxBBox));
    }
}

void OglVolumetricModel::updateVertexBuffer()
{
    const type::vector<Coord>& positions = m_positions.getValue();
    unsigned positionsBufferSize;

    positionsBufferSize = (positions.size()*sizeof(positions[0]));

    glBindBufferARB(GL_ARRAY_BUFFER, m_vbo);
    //Positions
    glBufferSubDataARB(GL_ARRAY_BUFFER,
        0,
        positionsBufferSize,
        positions.data());

    glBindBufferARB(GL_ARRAY_BUFFER, 0);

}

} // namespace sofa::component::visualmodel
