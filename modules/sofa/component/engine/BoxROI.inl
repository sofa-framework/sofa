/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_BOXROI_INL
#define SOFA_COMPONENT_ENGINE_BOXROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/BoxROI.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;

template <class DataTypes>
BoxROI<DataTypes>::BoxROI()
    : x0(new VecCoord)
    , boxes( initData( &boxes, "box", "DOFs in the box defined by xmin,ymin,zmin, xmax,ymax,zmax are fixed") )
    , f_X0( new XDataPtr<DataTypes>(&x0, "rest position coordinates of the degrees of freedom") )
    , f_indices( initData(&f_indices,"indices","Indices of the fixed points") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
{
    boxes.beginEdit()->push_back(Vec6(0,0,0,1,1,1));
    boxes.endEdit();

    addInput(f_X0);
    this->addField(f_X0,"rest_position");
    f_X0->init();

    addOutput(&f_indices);
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void BoxROI<DataTypes>::init()
{
    if (x0->empty())
    {
        BaseData* parent = mstate->findField("rest_position");
        f_X0->setParentValue(parent);
        parent->addOutput(f_X0);
        f_X0->setReadOnly(true);
        *x0 = f_X0->getValue();
    }
}


template <class DataTypes>
void BoxROI<DataTypes>::update()
{
    helper::vector<Vec6>& vb = *(boxes.beginEdit());
    SetIndex& indices = *(f_indices.beginEdit());

    for (unsigned int bi=0; bi<vb.size(); ++bi)
    {
        if (vb[bi][0] > vb[bi][3]) std::swap(vb[bi][0],vb[bi][3]);
        if (vb[bi][1] > vb[bi][4]) std::swap(vb[bi][1],vb[bi][4]);
        if (vb[bi][2] > vb[bi][5]) std::swap(vb[bi][2],vb[bi][5]);

        const Vec6& b=vb[bi];
        indices.clear();

        for( unsigned i=0; i<x0->size(); ++i )
        {
            Real x=0.0,y=0.0,z=0.0;
            DataTypes::get(x,y,z,(*x0)[i]);
            if( x >= b[0] && x <= b[3] && y >= b[1] && y <= b[4] && z >= b[2] && z <= b[5] )
            {
                indices.push_back(i);
            }
        }
    }

    f_indices.endEdit();
    boxes.endEdit();
}

template <class DataTypes>
BoxROI<DataTypes>::~BoxROI()
{}

template <class DataTypes>
void BoxROI<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *x0;

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        glColor4f (1,0.5,0.5,1);
        glDisable (GL_LIGHTING);
        glPointSize(10);

        glBegin (GL_POINTS);
        const SetIndex& indices = f_indices.getValue();
        for (typename SetIndex::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            gl::glVertexT(x[*it]);
        }
        glEnd();

        ///draw the constraint boxes
        glBegin(GL_LINES);
        const helper::vector<Vec6>& vb=boxes.getValue();
        for (unsigned int bi=0; bi<vb.size(); ++bi)
        {
            const Vec6& b=vb[bi];
            //const Vec6& b=box.getValue();
            const Real& Xmin=b[0];
            const Real& Xmax=b[3];
            const Real& Ymin=b[1];
            const Real& Ymax=b[4];
            const Real& Zmin=b[2];
            const Real& Zmax=b[5];
            glVertex3d(Xmin,Ymin,Zmin);
            glVertex3d(Xmin,Ymin,Zmax);
            glVertex3d(Xmin,Ymin,Zmin);
            glVertex3d(Xmax,Ymin,Zmin);
            glVertex3d(Xmin,Ymin,Zmin);
            glVertex3d(Xmin,Ymax,Zmin);
            glVertex3d(Xmin,Ymax,Zmin);
            glVertex3d(Xmax,Ymax,Zmin);
            glVertex3d(Xmin,Ymax,Zmin);
            glVertex3d(Xmin,Ymax,Zmax);
            glVertex3d(Xmin,Ymax,Zmax);
            glVertex3d(Xmin,Ymin,Zmax);
            glVertex3d(Xmin,Ymin,Zmax);
            glVertex3d(Xmax,Ymin,Zmax);
            glVertex3d(Xmax,Ymin,Zmax);
            glVertex3d(Xmax,Ymax,Zmax);
            glVertex3d(Xmax,Ymin,Zmax);
            glVertex3d(Xmax,Ymin,Zmin);
            glVertex3d(Xmin,Ymax,Zmax);
            glVertex3d(Xmax,Ymax,Zmax);
            glVertex3d(Xmax,Ymax,Zmin);
            glVertex3d(Xmax,Ymin,Zmin);
            glVertex3d(Xmax,Ymax,Zmin);
            glVertex3d(Xmax,Ymax,Zmax);
        }
        glEnd();
    }
    else // new drawing by spheres
    {
        glColor4f (1.0f,0.35f,0.35f,1.0f);
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        const SetIndex& indices = f_indices.getValue();
        for (typename SetIndex::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            helper::gl::drawSphere( x[*it], (float)_drawSize.getValue() );
        }
        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL);
    }
}

template <class DataTypes>
bool BoxROI<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    const helper::vector<Vec6>& vb=boxes.getValue();
    for (unsigned int bi=0; bi<vb.size(); ++bi)
    {
        const Vec6& b=vb[bi];
        //const Vec6& b=box.getValue();
        if (b[0] < minBBox[0]) minBBox[0] = b[0];
        if (b[1] < minBBox[1]) minBBox[1] = b[1];
        if (b[2] < minBBox[2]) minBBox[2] = b[2];
        if (b[3] > maxBBox[0]) maxBBox[0] = b[3];
        if (b[4] > maxBBox[1]) maxBBox[1] = b[4];
        if (b[5] > maxBBox[2]) maxBBox[2] = b[5];
    }
    return true;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
