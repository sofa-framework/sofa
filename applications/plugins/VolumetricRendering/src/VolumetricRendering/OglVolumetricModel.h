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
#ifndef OGLVOLUMETRICMODEL_H_
#define OGLVOLUMETRICMODEL_H_

#include <VolumetricRendering/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/visual/VisualModelImpl.h>
#include <sofa/gl/component/shader/OglVariable.h>
#include <sofa/gl/component/shader/OglAttribute.h>


namespace sofa
{
namespace component
{
namespace visualmodel
{

/**
 *  \brief Render 3D models with volume primitives (for now tetrahedron and hexahedron).
 *
 */

class SOFA_VOLUMETRICRENDERING_API OglVolumetricModel : public core::visual::VisualModel, public component::visual::Vec3State
{
public:
    SOFA_CLASS2(OglVolumetricModel, core::visual::VisualModel, component::visual::Vec3State);

    typedef sofa::core::topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::Hexahedron Hexahedron;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;

    Data< sofa::type::vector<Tetrahedron> > d_tetrahedra; ///< Tetrahedra to draw
    Data< sofa::type::vector<Hexahedron> > d_hexahedra; ///< Hexahedra to draw

    Data<float> d_volumeScale; ///< Scale for each volumetric primitive
    Data<bool> d_depthTest; ///< Set Depth Test
    Data<bool> d_blending; ///< Set Blending
    Data<type::Vec4f> d_defaultColor; ///< Color for each volume (if the attribute a_vertexColor is not detected)

    ~OglVolumetricModel() override;

protected:
    OglVolumetricModel();

private:
    core::topology::BaseMeshTopology::SPtr m_topology;
    gl::component::shader::OglShader::SPtr m_shader;

    bool b_modified;
    bool b_useTopology;

    GLuint m_vbo;
    bool b_tboCreated;
    GLuint m_tetraBarycentersTbo, m_tetraBarycentersTboTexture;
    GLuint m_hexaBarycentersTbo, m_hexaBarycentersTboTexture;

    void updateVertexBuffer();
    void splitHexahedra();
    void computeBarycenters();

    //Uniforms
    gl::component::shader::OglFloatVector4Variable::SPtr m_mappingTableValues;
    gl::component::shader::OglFloatVector4Variable::SPtr m_runSelectTableValues;

    //Attributes
    gl::component::shader::OglFloat4Attribute::SPtr m_vertexColors;

    sofa::type::vector<Tetrahedron> m_hexaToTetrahedra;

    sofa::type::vector<type::Vec3f> m_tetraBarycenters;
    sofa::type::vector<type::Vec3f> m_hexaBarycenters;

public:
    void init() override;
    void initVisual() override;
    void drawTransparent(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams *, bool onlyVisible=false) override;

    void handleTopologyChange() override;

    void updateVisual() override;
    void computeMeshFromTopology();

    bool insertInNode(core::objectmodel::BaseNode* node) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode(core::objectmodel::BaseNode* node) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

};

} // namespace visualmodel

} // namesapce component

} // namespace sofa


#endif // OGLVOLUMETRICMODEL_H_
