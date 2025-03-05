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
#pragma once
#include <sofa/component/engine/select/PairBoxRoi.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/BoundingBox.h>
#include <sofa/type/RGBAColor.h>
#include <limits>

namespace sofa::component::engine::select
{

template <class DataTypes>
PairBoxROI<DataTypes>::PairBoxROI()
    : inclusiveBox( initData(&inclusiveBox, "inclusiveBox", "Inclusive box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , includedBox( initData(&includedBox, "includedBox", "Included box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , positions(initData(&positions,"meshPosition","Vertices of the mesh loaded"))
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_pointsInROI( initData(&f_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , p_drawInclusiveBox( initData(&p_drawInclusiveBox,false,"drawInclusiveBox","Draw Inclusive Box") )
    , p_drawIncludedBox( initData(&p_drawIncludedBox,false,"drawIncludedBox","Draw Included Box") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points") )
    , _drawSize( initData(&_drawSize,"drawSize","Draw Size") )
{
    //Adding alias to handle old PairBoxROI input/output
    addAlias(&f_pointsInROI,"pointsInBox");
    addAlias(&f_X0,"rest_position");

    addInput(&f_X0);

    addOutput(&f_indices);
    addOutput(&f_pointsInROI);
}

template <class DataTypes>
void PairBoxROI<DataTypes>::init()
{
    using sofa::core::objectmodel::BaseData;
    using sofa::core::objectmodel::BaseContext;

    if (!f_X0.isSet())
    {
        sofa::core::behavior::BaseMechanicalState* mstate;
        this->getContext()->get(mstate,BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = nullptr;
            this->getContext()->get(loader,BaseContext::Local);
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    f_X0.setParent(parent);
                    f_X0.setReadOnly(true);
                }
            }
            else   // no local state, no loader => find upward
            {
                this->getContext()->get(mstate,BaseContext::SearchUp);
                assert(mstate && "PairBoxROI needs a mstate");
                BaseData* parent = mstate->findData("rest_position");
                assert(parent && "PairBoxROI needs a state with a rest_position Data");
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
    }

    setDirtyValue();

}

template <class DataTypes>
void PairBoxROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
bool PairBoxROI<DataTypes>::isPointInBox(const typename DataTypes::CPos& p, const Vec6& b)
{
    return ( p[0] >= b[0] && p[0] <= b[3] && p[1] >= b[1] && p[1] <= b[4] && p[2] >= b[2] && p[2] <= b[5] );
}

template <class DataTypes>
bool PairBoxROI<DataTypes>::isPointInBox(const PointID& pid, const Vec6& b)
{
    const VecCoord* x0 = &f_X0.getValue();
    CPos p =  DataTypes::getCPos((*x0)[pid]);

    return ( isPointInBox(p,b) );
}

template <class DataTypes>
void PairBoxROI<DataTypes>::doUpdate()
{
   const VecCoord* x0 = &f_X0.getValue();

   Vec6& maxvb = *(inclusiveBox.beginEdit());
   Vec6& minvb = *(includedBox.beginEdit());

    if (maxvb==Vec6(0,0,0,0,0,0)|| minvb==Vec6(0,0,0,0,0,0))
    {
        inclusiveBox.endEdit();
        includedBox.endEdit();
        return;
    }


    if (maxvb[0] > maxvb[3]) std::swap(maxvb[0],maxvb[3]);
    if (maxvb[1] > maxvb[4]) std::swap(maxvb[1],maxvb[4]);
    if (maxvb[2] > maxvb[5]) std::swap(maxvb[2],maxvb[5]);
    
    if (minvb[0] > minvb[3]) std::swap(minvb[0],minvb[3]);
    if (minvb[1] > minvb[4]) std::swap(minvb[1],minvb[4]);
    if (minvb[2] > minvb[5]) std::swap(minvb[2],minvb[5]);
    
    inclusiveBox.endEdit();



    // Write accessor for topological element indices in BOX
    SetIndex& indices = *f_indices.beginWriteOnly();
   
    // Write accessor for toplogical element in BOX
    helper::WriteOnlyAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;

    // Clear lists
    indices.clear();
   
    pointsInROI.clear();


    //Points
    for(size_t i=0; i<x0->size(); ++i )
    {
        if (isPointInBox(i,maxvb) && !isPointInBox(i,minvb))
        {
            indices.push_back(i);
            pointsInROI.push_back((*x0)[i]);
        }
    }
   
    f_indices.endEdit();
    
}


template <class DataTypes>
void PairBoxROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels() && !this->_drawSize.getValue())
        return;

    constexpr sofa::type::RGBAColor color(1.0f, 0.4f, 0.4f, 1.0f);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    /// Draw inclusive box
    if( p_drawInclusiveBox.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        const float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        const Vec6& vb=inclusiveBox.getValue();
        const sofa::type::Vec3 minBBox(vb[0], vb[1], vb[2]);
        const sofa::type::Vec3 maxBBox(vb[3], vb[4], vb[5]);
        vparams->drawTool()->setMaterial(color);
        vparams->drawTool()->drawBoundingBox(minBBox, maxBBox, linesWidth);
    }

    /// Draw included box
    if(p_drawIncludedBox.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        const float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        const Vec6& vb=includedBox.getValue();
        const sofa::type::Vec3 minBBox(vb[0], vb[1], vb[2]);
        const sofa::type::Vec3 maxBBox(vb[3], vb[4], vb[5]);
        vparams->drawTool()->setMaterial(color);
        vparams->drawTool()->drawBoundingBox(minBBox, maxBBox, linesWidth);
    }

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    /// Draw points in ROI
    if( p_drawPoints.getValue())
    {
        const float pointsWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<sofa::type::Vec3> vertices;
        helper::ReadAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            sofa::type::Vec3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawPoints(vertices, pointsWidth, color);
    }


}

} //namespace sofa::component::engine::select
