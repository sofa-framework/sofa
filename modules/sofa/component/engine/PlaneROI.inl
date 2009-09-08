/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_PLANEROI_INL
#define SOFA_COMPONENT_ENGINE_PLANEROI_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/PlaneROI.h>
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
PlaneROI<DataTypes>::PlaneROI()
    : planes( initData(&planes, "plane", "Plane defined by 3 points and a depth distance") )
    , f_X0( initData (&f_X0, "rest_position", "Rest position coordinates of the degrees of freedom") )
    , f_indices( initData(&f_indices,"indices","Indices of the points contained in the ROI") )
    , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering") )
{
    planes.beginEdit()->push_back(Vec10(Vec<9,Real>(0,0,0,0,0,0,0,0,0),0));
    planes.endEdit();

    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}

template <class DataTypes>
void PlaneROI<DataTypes>::init()
{
    if (!f_X0.isSet())
    {
        MechanicalState<DataTypes>* mstate;
        this->getContext()->get(mstate);
        if (mstate)
        {
            BaseData* parent = mstate->findField("rest_position");
            if (parent)
            {
                f_X0.setParentValue(parent);
                parent->addOutput(&f_X0);
                f_X0.setReadOnly(true);
            }
        }
    }
    addInput(&f_X0);
    addOutput(&f_indices);
    setDirtyValue();
}

template <class DataTypes>
void PlaneROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void PlaneROI<DataTypes>::update()
{
    cleanDirty();

    SetIndex& indices = *(f_indices.beginEdit());
    indices.clear();

    const VecCoord* x0 = &f_X0.getValue();

    for( unsigned i=0; i<x0->size(); ++i )
    {
        Real x=0.0,y=0.0,z=0.0;
        DataTypes::get(x,y,z,(*x0)[i]);

        Vec3 pt = Vec3(x,y,z);

        const helper::vector<Vec10>& vp=planes.getValue();
        for (unsigned int pi=0; pi<vp.size(); ++pi)
        {
            const Vec10& p=vp[pi];
            Vec3 p0 = Vec3(p[0], p[1], p[2]);
            Vec3 p1 = Vec3(p[3], p[4], p[5]);
            Vec3 p2 = Vec3(p[6], p[7], p[8]);
            const Real depth = p[9];

            Vec3 vdepth = (p1-p0).cross(p2-p0);
            vdepth.normalize();

            Vec3 p3 = p0 + (p2-p1);
            Vec3 p4 = p0 + vdepth * (depth/2);
            Vec3 p6 = p2 + vdepth * (depth/2);

            Vec3 plane0 = (p1-p0).cross(p4-p0);
            plane0.normalize();

            Vec3 plane1 = (p2-p3).cross(p6-p3);
            plane1.normalize();

            Vec3 plane2 = (p3-p0).cross(p4-p0);
            plane2.normalize();

            Vec3 plane3 = (p2-p1).cross(p6-p2);
            plane3.normalize();

            Real width = fabs(dot((p2-p0),plane0));
            Real length = fabs(dot((p2-p0),plane2));

            Vec3 pv0 = (pt-p0);
            Vec3 pv1 = (pt-p2);

            if( fabs(dot(pv0, plane0)) <= width && fabs(dot(pv1, plane1)) <= width )
            {
                if ( fabs(dot(pv0, plane2)) <= length && fabs(dot(pv1, plane3)) <= length )
                {
                    if ( fabs(dot(pv0, vdepth)) <= fabs(depth/2) )
                    {
                        indices.push_back(i);
                        break;
                    }
                }
            }
        }
    }
    f_indices.endEdit();
}

template <class DataTypes>
void PlaneROI<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels())
        return;

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        ///draw the boxes
        glBegin(GL_LINES);
        const helper::vector<Vec10>& vp=planes.getValue();
        for (unsigned int pi=0; pi<vp.size(); ++pi)
        {
            const Vec10& p=vp[pi];
            Vec3 p0 = Vec3(p[0], p[1], p[2]);
            Vec3 p1 = Vec3(p[3], p[4], p[5]);
            Vec3 p2 = Vec3(p[6], p[7], p[8]);

            Vec3 p3 = p0 + (p2 - p1);

            Vec3 n = (p1-p0).cross(p2-p0);
            n.normalize();
            const Real depth = p[9];

            Vec3 p4 = p0 + n * (depth/2);
            p0 = p0 + (-n) * (depth/2);
            Vec3 p5 = p1 + n * (depth/2);
            p1 = p1 + (-n) * (depth/2);
            Vec3 p6 = p2 + n * (depth/2);
            p2 = p2 + (-n) * (depth/2);
            Vec3 p7 = p3 + n * (depth/2);
            p3 = p3 + (-n) * (depth/2);

            glVertex3d(p0.x(), p0.y(), p0.z());
            glVertex3d(p1.x(), p1.y(), p1.z());
            glVertex3d(p1.x(), p1.y(), p1.z());
            glVertex3d(p2.x(), p2.y(), p2.z());
            glVertex3d(p2.x(), p2.y(), p2.z());
            glVertex3d(p3.x(), p3.y(), p3.z());
            glVertex3d(p3.x(), p3.y(), p3.z());
            glVertex3d(p0.x(), p0.y(), p0.z());

            glVertex3d(p4.x(), p4.y(), p4.z());
            glVertex3d(p5.x(), p5.y(), p5.z());
            glVertex3d(p5.x(), p5.y(), p5.z());
            glVertex3d(p6.x(), p6.y(), p6.z());
            glVertex3d(p6.x(), p6.y(), p6.z());
            glVertex3d(p7.x(), p7.y(), p7.z());
            glVertex3d(p7.x(), p7.y(), p7.z());
            glVertex3d(p4.x(), p4.y(), p4.z());

            glVertex3d(p0.x(), p0.y(), p0.z());
            glVertex3d(p4.x(), p4.y(), p4.z());

            glVertex3d(p1.x(), p1.y(), p1.z());
            glVertex3d(p5.x(), p5.y(), p5.z());

            glVertex3d(p2.x(), p2.y(), p2.z());
            glVertex3d(p6.x(), p6.y(), p6.z());

            glVertex3d(p3.x(), p3.y(), p3.z());
            glVertex3d(p7.x(), p7.y(), p7.z());
        }
        glEnd();
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
