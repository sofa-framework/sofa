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
#ifndef SOFA_COMPONENT_VISUALMODEL_DATADISPLAY_H
#define SOFA_COMPONENT_VISUALMODEL_DATADISPLAY_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaOpenglVisual/OglColorMap.h>
#include <SofaBaseVisual/VisualModelImpl.h>

#include <sofa/defaulttype/RGBAColor.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API DataDisplay : public core::visual::VisualModel, public ExtVec3fState
{
public:
    SOFA_CLASS2(DataDisplay, core::visual::VisualModel, ExtVec3fState);

    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad     Quad;

    typedef helper::vector<Real> VecPointData;
    typedef helper::vector<Real> VecCellData;

public:
    Data<bool> f_maximalRange; ///< Keep the maximal range through all timesteps
    Data<VecPointData> f_pointData; ///< Data associated with nodes
    Data<VecCellData> f_triangleData; ///< Data associated with triangles
    Data<VecCellData> f_quadData; ///< Data associated with quads
    Data<VecPointData> f_pointTriangleData; ///< Data associated with nodes per triangle
    Data<VecPointData> f_pointQuadData; ///< Data associated with nodes per quad
    Data<defaulttype::RGBAColor> f_colorNaN; ///< Color for NaNs
    Data<defaulttype::Vec2f> d_userRange; ///< Clamp to this values (if max>min)
    Data<float> d_currentMin; ///< Current min range
    Data<float> d_currentMax; ///< Current max range
    Data<float> d_shininess; ///< Shininess for rendering point-based data [0,128].  <0 means no specularity

    visualmodel::OglColorMap *colorMap;
    core::State<DataTypes> *state;
    core::topology::BaseMeshTopology* topology;
    Real oldMin, oldMax;

    void init() override;
    void drawVisual(const core::visual::VisualParams* vparams) override;
    void updateVisual() override;

    virtual bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    virtual bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

protected:
    void computeNormals();
    helper::vector<defaulttype::Vec3f> m_normals;

    DataDisplay()
        : f_maximalRange(initData(&f_maximalRange, true, "maximalRange", "Keep the maximal range through all timesteps"))
          , f_pointData(initData(&f_pointData, "pointData", "Data associated with nodes"))
          , f_triangleData(initData(&f_triangleData, "triangleData", "Data associated with triangles"))
          , f_quadData(initData(&f_quadData, "quadData", "Data associated with quads"))
          , f_pointTriangleData(initData(&f_pointTriangleData, "pointTriangleData", "Data associated with nodes per triangle"))
          , f_pointQuadData(initData(&f_pointQuadData, "pointQuadData", "Data associated with nodes per quad"))
          , f_colorNaN(initData(&f_colorNaN, defaulttype::RGBAColor(0.0f,0.0f,0.0f,1.0f), "colorNaN", "Color used for NaN values.(default=[0.0,0.0,0.0,1.0])"))
          , d_userRange(initData(&d_userRange, defaulttype::Vec2f(1,-1), "userRange", "Clamp to this values (if max>min)"))
          , d_currentMin(initData(&d_currentMin, 0.f, "currentMin", "Current min range"))
          , d_currentMax(initData(&d_currentMax, 0.f, "currentMax", "Current max range"))
          , d_shininess(initData(&d_shininess, -1.f, "shininess", "Shininess for rendering point-based data [0,128].  <0 means no specularity"))
          , state(NULL)
          , topology(NULL)
          , oldMin(0)
          , oldMax(0)
    {
        this->addAlias(&f_triangleData,"cellData"); // backward compatibility
        d_currentMin.setReadOnly(true);
        d_currentMax.setReadOnly(true);
    }
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // #ifndef SOFA_COMPONENT_VISUALMODEL_DATADISPLAY_H
