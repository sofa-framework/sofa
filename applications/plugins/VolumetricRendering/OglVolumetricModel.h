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
#ifndef OGLVOLUMETRICMODEL_H_
#define OGLVOLUMETRICMODEL_H_

#include <VolumetricRendering/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaOpenglVisual/OglVariable.h>

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

class SOFA_VOLUMETRICRENDERING_API OglVolumetricModel : public core::visual::VisualModel, public ExtVec3fState
{
public:
    SOFA_CLASS2(OglVolumetricModel, core::visual::VisualModel, ExtVec3fState);

    typedef sofa::core::topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::Hexahedron Hexahedron;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;

    Data< sofa::defaulttype::ResizableExtVector<Tetrahedron> > d_tetrahedra;
    Data< sofa::defaulttype::ResizableExtVector<Hexahedron> > d_hexahedra;

    Data<bool> d_depthTest;
    Data<bool> d_blending;

    virtual ~OglVolumetricModel();

protected:
    OglVolumetricModel();

private:
    core::topology::BaseMeshTopology::SPtr m_topology;

    bool b_modified;
    bool b_useTopology;

    GLuint m_vbo;
    void updateVertexBuffer();
    void splitHexahedra();

    //Tables
    sofa::component::visualmodel::OglFloatVector4Variable::SPtr m_mappingTableValues;
    sofa::component::visualmodel::OglFloatVector4Variable::SPtr m_runSelectTableValues;

    sofa::defaulttype::ResizableExtVector<Tetrahedron> m_hexaToTetrahedra;

public:
    void init();
    void initVisual();
    void drawTransparent(const core::visual::VisualParams* vparams);
    void computeBBox(const core::ExecParams *, bool onlyVisible=false);

    void handleTopologyChange();

    void updateVisual();
    void computeMeshFromTopology();

    bool insertInNode(core::objectmodel::BaseNode* node) { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode(core::objectmodel::BaseNode* node) { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

};

} // namespace visualmodel

} // namesapce component

} // namespace sofa


#endif // OGLVOLUMETRICMODEL_H_
