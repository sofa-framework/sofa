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
#ifndef SOFA_COMPONENT_VISUALMODEL_DATADISPLAY_H
#define SOFA_COMPONENT_VISUALMODEL_DATADISPLAY_H

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaOpenglVisual/ColorMap.h>
#include <SofaBaseVisual/VisualModelImpl.h>

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

protected:

    DataDisplay()
        : f_maximalRange(initData(&f_maximalRange, true, "maximalRange", "Keep the maximal range through all timesteps"))
          , f_pointData(initData(&f_pointData, "pointData", "Data associated with nodes"))
          , f_cellData(initData(&f_cellData, "cellData", "Data associated with elements"))
          , f_colorNaN(initData(&f_colorNaN, sofa::defaulttype::Vec4f(0.0f,0.0f,0.0f,1.0f), "colorNaN", "Color used for NaN values"))
          , state(NULL)
          , topology(NULL)
          , oldMin(0)
          , oldMax(0)
    {}

public:

    Data<bool> f_maximalRange;
    Data<VecPointData> f_pointData;
    Data<VecCellData> f_cellData;
    Data<sofa::defaulttype::Vec4f> f_colorNaN; // Color for NaNs (alpha channel is not used)

    visualmodel::ColorMap *colorMap;
    core::State<DataTypes> *state;
    core::topology::BaseMeshTopology* topology;
    Real oldMin, oldMax;

    void init();
    //void reinit();

    //void initVisual() { initTextures(); }
    //void clearVisual() { }
    //void initTextures() {}
    void drawVisual(const core::visual::VisualParams* vparams);
    //void drawTransparent(const VisualParams* /*vparams*/)
    void updateVisual();

protected:

    void computeNormals();
    helper::vector<defaulttype::Vec3f> m_normals;

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif // #ifndef SOFA_COMPONENT_VISUALMODEL_DATADISPLAY_H
