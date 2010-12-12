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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define FRAME_FRAMEMECHANICALOBJECT_CPP

#include "FrameMechanicalObject.h"
#include <sofa/component/container/MechanicalObject.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace container
{


template <>
void MechanicalObject<Affine3dTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels()) return;
    typedef Vec<3,double> Vec3d;
    typedef Vec<4,float> Vec4f;
    std::vector<Vec3d> points;
//                cerr<<"MechanicalObject<Affine3dTypes>::draw()"<<endl;
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(10);
    for(int i=0; i<this->getSize(); i++ )
    {
        const Affine3dTypes::Coord& c = (*getX())[i];
//                    cerr<<"MechanicalObject<Affine3dTypes>::draw, c.getCenter() = " << c.getCenter() << endl;
//                    cerr<<"MechanicalObject<Affine3dTypes>::draw, c.gAffine() = " << c.getMaterialFrame() << endl;
        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+c.getAffine()[0][0], c.getCenter()[1]+c.getAffine()[1][0], c.getCenter()[2]+c.getAffine()[2][0] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(1,0,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+c.getAffine()[0][1], c.getCenter()[1]+c.getAffine()[1][1], c.getCenter()[2]+c.getAffine()[2][1] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,1,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back( Vec3d(c.getCenter()[0]+c.getAffine()[0][2], c.getCenter()[1]+c.getAffine()[1][2], c.getCenter()[2]+c.getAffine()[2][2] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,0,1,1));

    }
    glPopAttrib();
}


template <>
void MechanicalObject<Quadratic3dTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels()) return;
    typedef Vec<3,double> Vec3d;
    typedef Vec<4,float> Vec4f;
    std::vector<Vec3d> points;
//                cerr<<"MechanicalObject<Quadratic3dTypes>::draw()"<<endl;
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(10);
    for(int i=0; i<this->getSize(); i++ )
    {
        const Quadratic3dTypes::Coord& c = (*getX())[i];
        points.clear();
        Quadratic3dTypes::Affine aff = c.getAffine();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+aff[0][0], c.getCenter()[1]+aff[1][0], c.getCenter()[2]+aff[2][0] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(1,0,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+aff[0][1], c.getCenter()[1]+aff[1][1], c.getCenter()[2]+aff[2][1] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,1,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back( Vec3d(c.getCenter()[0]+aff[0][2], c.getCenter()[1]+aff[1][2], c.getCenter()[2]+aff[2][2] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,0,1,1));

    }
    glPopAttrib();
}


template <>
void MechanicalObject<DeformationGradient331dTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels()) return;
    typedef Vec<3,double> Vec3d;
    typedef Vec<4,float> Vec4f;
    std::vector<Vec3d> points;
//                cerr<<"MechanicalObject<Affine3dTypes>::draw()"<<endl;
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(10);
    for(int i=0; i<this->getSize(); i++ )
    {
        const DeformationGradient331dTypes::Coord& c = (*getX())[i];
//                    cerr<<"MechanicalObject<Affine3dTypes>::draw, c.getCenter() = " << c.getCenter() << endl;
//                    cerr<<"MechanicalObject<Affine3dTypes>::draw, c.gAffine() = " << c.getMaterialFrame() << endl;
        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+c.getMaterialFrame()[0][0], c.getCenter()[1]+c.getMaterialFrame()[1][0], c.getCenter()[2]+c.getMaterialFrame()[2][0] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(1,0,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+c.getMaterialFrame()[0][1], c.getCenter()[1]+c.getMaterialFrame()[1][1], c.getCenter()[2]+c.getMaterialFrame()[2][1] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,1,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back( Vec3d(c.getCenter()[0]+c.getMaterialFrame()[0][2], c.getCenter()[1]+c.getMaterialFrame()[1][2], c.getCenter()[2]+c.getMaterialFrame()[2][2] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,0,1,1));

    }
    glPopAttrib();
}

template <>
void MechanicalObject<DeformationGradient332dTypes >::draw()
{
    if (!this->getContext()->getShowBehaviorModels()) return;
    typedef Vec<3,double> Vec3d;
    typedef Vec<4,float> Vec4f;
    std::vector<Vec3d> points;
//                cerr<<"MechanicalObject<Affine3dTypes>::draw()"<<endl;
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(10);
    for(int i=0; i<this->getSize(); i++ )
    {
        const DeformationGradient332dTypes::Coord& c = (*getX())[i];
//                    cerr<<"MechanicalObject<Affine3dTypes>::draw, c.getCenter() = " << c.getCenter() << endl;
//                    cerr<<"MechanicalObject<Affine3dTypes>::draw, c.gAffine() = " << c.getMaterialFrame() << endl;
        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+c.getMaterialFrame()[0][0], c.getCenter()[1]+c.getMaterialFrame()[1][0], c.getCenter()[2]+c.getMaterialFrame()[2][0] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(1,0,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back(Vec3d( c.getCenter()[0]+c.getMaterialFrame()[0][1], c.getCenter()[1]+c.getMaterialFrame()[1][1], c.getCenter()[2]+c.getMaterialFrame()[2][1] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,1,0,1));

        points.clear();
        points.push_back(Vec3d(c.getCenter()[0], c.getCenter()[1], c.getCenter()[2] ));
        points.push_back( Vec3d(c.getCenter()[0]+c.getMaterialFrame()[0][2], c.getCenter()[1]+c.getMaterialFrame()[1][2], c.getCenter()[2]+c.getMaterialFrame()[2][2] ));
        simulation::getSimulation()->DrawUtility.drawLines(points,2,Vec4d(0,0,1,1));

    }
    glPopAttrib();
}



SOFA_DECL_CLASS(FrameMechanicalObject)

using namespace sofa::defaulttype;

int MechanicalObjectClass = core::RegisterObject("mechanical state vectors")
#ifndef SOFA_FLOAT
        .add< MechanicalObject<Affine3dTypes> >()
        .add< MechanicalObject<Quadratic3dTypes> >()
        .add< MechanicalObject<DeformationGradient331dTypes> >()
        .add< MechanicalObject<DeformationGradient332dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<Affine3fTypes> >()
        .add< MechanicalObject<Quadratic3fTypes> >()
        .add< MechanicalObject<DeformationGradient331fTypes> >()
        .add< MechanicalObject<DeformationGradient332fTypes> >()
#endif
        ;



#ifndef SOFA_FLOAT
template class SOFA_FRAME_API MechanicalObject<Affine3dTypes>;
template class SOFA_FRAME_API MechanicalObject<Quadratic3dTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient331dTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient332dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API MechanicalObject<Affine3fTypes>;
template class SOFA_FRAME_API MechanicalObject<Quadratic3fTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient331fTypes>;
template class SOFA_FRAME_API MechanicalObject<DeformationGradient332fTypes>;
#endif
} // namespace behavior

} // namespace core

} // namespace sofa
