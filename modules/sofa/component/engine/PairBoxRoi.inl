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
#ifndef SOFA_COMPONENT_ENGINE_PAIRBOXROI_INL
#define SOFA_COMPONENT_ENGINE_PAIRBOXROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/PairBoxRoi.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;
using namespace core::topology;

template <class DataTypes>
PairBoxROI<DataTypes>::PairBoxROI()
    : inclusiveBox( initData(&inclusiveBox, "inclusiveBox", "Inclusive box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , includedBox( initData(&includedBox, "includedBox", "Included box defined by xmin,ymin,zmin, xmax,ymax,zmax") )
    , f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , f_pointsInROI( initData(&f_pointsInROI,"pointsInROI","Points contained in the ROI") )
    , p_drawInclusiveBox( initData(&p_drawInclusiveBox,false,"drawInclusiveBox","Draw Inclusive Box") )
    , p_drawIncludedBox( initData(&p_drawIncludedBox,false,"drawInclusdedBx","Draw Included Box") )
    , p_drawPoints( initData(&p_drawPoints,false,"drawPoints","Draw Points") )
    , positions(initData(&positions,"meshPosition","Vertices of the mesh loaded"))
    ,p_cornerPoints(initData(&p_cornerPoints,"cornerPoints","Corner positions for bilinear constraint"))
{
    //Adding alias to handle old PairBoxROI input/output
    addAlias(&f_pointsInROI,"pointsInBox");
    addAlias(&f_X0,"rest_position");

    inclusiveBox.beginEdit();
    inclusiveBox=Vec6(0,0,0,1,1,1);
    inclusiveBox.endEdit();

    includedBox.beginEdit();
    includedBox=Vec6(0.2,0.2,0.2,0.8,0.8,0.8);
    includedBox.endEdit();
  
}

template <class DataTypes>
void PairBoxROI<DataTypes>::init()
{
    if (!f_X0.isSet())
    {
        BaseMechanicalState* mstate;
        this->getContext()->get(mstate,BaseContext::Local);
        if (mstate)
        {
            BaseData* parent = mstate->findField("rest_position");
            if (parent)
            {
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = NULL;
            this->getContext()->get(loader,BaseContext::Local);
            if (loader)
            {
                BaseData* parent = loader->findField("position");
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
                BaseData* parent = mstate->findField("rest_position");
                assert(parent && "PairBoxROI needs a state with a rest_position Data");
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
    }

    // Find the 4 corners of the grid topology
    Coord corner0, corner1, corner2, corner3;
    helper::ReadAccessor<Data<VecCoord> > raPositions = positions;
    std::cout << "raPositions.size() = " << raPositions.size() << std::endl;
    if(raPositions.size() > 0)
    {
        corner0 = raPositions[0];
        corner1 = raPositions[0];
        corner2 = raPositions[0];
        corner3 = raPositions[0];
        for (size_t i = 0; i < raPositions.size() ; i++)
        {
            if(raPositions[i][0] < corner0[0] || raPositions[i][1] < corner0[1] || raPositions[i][2] < corner0[2])
            {
                corner0 = raPositions[i];
            }

            if(raPositions[i][0] > corner2[0] || raPositions[i][1] > corner2[1] || raPositions[i][2] > corner2[2])
            {   
                 corner2 = raPositions[i];
            }

            if(raPositions[i][1] < corner1[1] || raPositions[i][0] > corner1[0] )
            {   
                 corner1 = raPositions[i];
            }

            else if(raPositions[i][0] < corner3[0] || raPositions[i][1] > corner3[1])
            {   
                 corner3 = raPositions[i];
            }
         }

        // epsilon should be a data 
        Vec<3,SReal> epsilon(0.1,0.1,0.1);
        for(int i = 0; i<3 ; ++i)
        {
            if(corner0[i] == corner2[i])
                epsilon[i] = 0;
        }

        // Define the inclusive and included box
        inclusiveBox.beginEdit();
        inclusiveBox=Vec6(corner0[0]-epsilon[0],corner0[1]-epsilon[1],corner0[2]-epsilon[2],corner2[0] + epsilon[0],corner2[1] + epsilon[1],corner2[2] +epsilon[2]);
        inclusiveBox.endEdit();

        includedBox.beginEdit();
        includedBox=Vec6(corner0[0]+epsilon[0],corner0[1]+epsilon[1],corner0[2]+epsilon[2],corner2[0]-epsilon[0],corner2[1]-epsilon[1],corner2[2]-epsilon[2]);
        includedBox.endEdit();

        // set corner positions
        // Write accessor 
        helper::WriteAccessor< Data<VecCoord > > cornerPositions = p_cornerPoints;

        cornerPositions.push_back(corner0);
        cornerPositions.push_back(corner1);
        cornerPositions.push_back(corner2);
        cornerPositions.push_back(corner3);
    }

 
    addInput(&f_X0);

    addOutput(&f_indices);
    addOutput(&f_pointsInROI);
    addOutput(&p_cornerPoints);
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
void PairBoxROI<DataTypes>::update()
{
    cleanDirty();

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
    SetIndex& indices = *f_indices.beginEdit();
   
    // Write accessor for toplogical element in BOX
    helper::WriteAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;

    // Clear lists
    indices.clear();
   
    pointsInROI.clear();

    const VecCoord* x0 = &f_X0.getValue();

    //Points
    std::cout << "x0->size() = " << x0->size() << std::endl;
    for( unsigned i=0; i<x0->size(); ++i )
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

    const VecCoord* x0 = &f_X0.getValue();
    sofa::defaulttype::Vec4f color = sofa::defaulttype::Vec4f(1.0f, 0.4f, 0.4f, 1.0f);


    /// Draw inclusive box
    if( p_drawInclusiveBox.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<sofa::defaulttype::Vector3> vertices;
        const Vec6& vb=inclusiveBox.getValue();

        const Vec6& b=vb;
        const Real& Xmin=b[0];
        const Real& Xmax=b[3];
        const Real& Ymin=b[1];
        const Real& Ymax=b[4];
        const Real& Zmin=b[2];
        const Real& Zmax=b[5];
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmax) );
        vparams->drawTool()->drawLines(vertices, linesWidth , color );
    }

    /// Draw included box
    if(p_drawIncludedBox.getValue())
    {
        vparams->drawTool()->setLightingEnabled(false);
        float linesWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        std::vector<sofa::defaulttype::Vector3> vertices;
        const Vec6& vb=includedBox.getValue();

        const Vec6& b=vb;
        const Real& Xmin=b[0];
        const Real& Xmax=b[3];
        const Real& Ymin=b[1];
        const Real& Ymax=b[4];
        const Real& Zmin=b[2];
        const Real& Zmax=b[5];
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmin,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmax) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymin,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmin) );
        vertices.push_back( sofa::defaulttype::Vector3(Xmax,Ymax,Zmax) );
        vparams->drawTool()->drawLines(vertices, linesWidth , color );
    }

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    /// Draw points in ROI
    if( p_drawPoints.getValue())
    {
        float pointsWidth = _drawSize.getValue() ? (float)_drawSize.getValue() : 1;
        vparams->drawTool()->setLightingEnabled(false);
        std::vector<sofa::defaulttype::Vector3> vertices;
        helper::ReadAccessor< Data<VecCoord > > pointsInROI = f_pointsInROI;
        for (unsigned int i=0; i<pointsInROI.size() ; ++i)
        {
            CPos p = DataTypes::getCPos(pointsInROI[i]);
            sofa::defaulttype::Vector3 pv;
            for( unsigned int j=0 ; j<max_spatial_dimensions ; ++j )
                pv[j] = p[j];
            vertices.push_back( pv );
        }
        vparams->drawTool()->drawPoints(vertices, pointsWidth, color);
    }

}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
