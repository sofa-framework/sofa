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
#ifndef OGLTETRAHEDRALMODEL_H_
#define OGLTETRAHEDRALMODEL_H_

#include <VolumetricRendering/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/gl/component/shader/OglVariable.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

/**
 *  \brief Render 3D models with tetrahedra.
 *
 *  This is a basic class using tetrehedra for the rendering
 *  instead of common triangles. It loads its data with
 *  a BaseMeshTopology and a MechanicalState.
 *  This rendering is only available with Nvidia's >8 series
 *  and Ati's >2K series.
 *
 */

template<class DataTypes>
class OglTetrahedralModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglTetrahedralModel, core::visual::VisualModel);

    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::core::topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    core::topology::BaseMeshTopology* m_topology;

    core::topology::PointData< sofa::type::vector<Coord> > m_positions; ///< Vertices coordinates
    Data< sofa::type::vector<Tetrahedron> > m_tetrahedrons;
    Data<bool> depthTest; ///< Set Depth Test
    Data<bool> blending; ///< Set Blending

    bool modified;
    int lastMeshRev;
    bool useTopology;

private:
    GLuint m_vbo;
    void updateVertexBuffer();

    //Tables
    sofa::gl::component::shader::OglFloatVector4Variable::SPtr m_mappingTableValues;
    sofa::gl::component::shader::OglFloatVector4Variable::SPtr m_runSelectTableValues;

protected:
    OglTetrahedralModel();
    ~OglTetrahedralModel() override;
public:
    void init() override;
    void initVisual() override;
    void drawTransparent(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams *, bool onlyVisible=false) override;

    void updateVisual() override;
    virtual void computeMesh();
};

#if  !defined(SOFA_COMPONENT_VISUALMODEL_OGLTETRAHEDRALMODEL_CPP)
extern template class SOFA_VOLUMETRICRENDERING_API OglTetrahedralModel<defaulttype::Vec3Types>;

#endif

} // namespace visualmodel

} // namesapce component

} // namespace sofa


#endif /*OGLTETRAHEDRALMODEL_H_*/
