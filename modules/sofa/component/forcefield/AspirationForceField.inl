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
#ifndef SOFA_COMPONENT_FORCEFIELD_ASPIRATEFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_ASPIRATEFORCEFIELD_INL

#include "AspirationForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
//#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/behavior/ForceField.inl>



namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
AspirationForceField<DataTypes>::AspirationForceField()
    : f_force(initData(&f_force, "force", "applied force to all points if forces attribute is not specified"))
    , f_scale(initData(&f_scale, "force_scale", "scale the applied force"))
    , f_arrowSizeCoef(initData(&f_arrowSizeCoef,0.0, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
    , f_ray(initData(&f_ray, "ray", "total force for all points, will be distributed uniformly over points"))
    , f_positionRigid(initData(&f_positionRigid, "position", "Position's rigid coord"))
    , f_opposite_pressure(initData(&f_opposite_pressure,0.0, "opposite_pressure", "opposite_pressure"))


{
}


template<class DataTypes>
void AspirationForceField<DataTypes>::addForce(DataVecDeriv& f1, const DataVecCoord& p1, const DataVecDeriv&, const core::MechanicalParams* /*params*/)
{
    const VecCoord & x = *this->mstate->getX();
    RigidCoord position = f_positionRigid.getValue();

    Coord center = position.getCenter();
    Deriv force = f_force.getValue();
    Coord orientation = position.rotate(f_force.getValue());

    double scale = f_scale.getValue();
    double ray = f_ray.getValue();

    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > _f1 = f1;
    _f1.resize(p1.getValue().size());

    for (unsigned int i=0; i<x.size(); i++)
    {
        Coord dir = x[i]-center;
        double norm = dir.norm();
        if (norm < ray)
        {
            if (dir*orientation>=0)   //
            {
                _f1[i] -= (dir*(1.0-(norm/ray)) * scale);
            }
            else if (f_opposite_pressure.getValue()>0.0)
            {
                _f1[i] -= (orientation*f_opposite_pressure.getValue());
            }
        }
    }
}

template<class DataTypes>
void AspirationForceField<DataTypes>::addDForce(DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */, const core::MechanicalParams* mparams)
{
    //TODO: remove this line (avoid warning message) ...
    mparams->kFactor();
};

template<class DataTypes>
void AspirationForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix */*mat*/, SReal /*k*/, unsigned int &/*offset*/)
{

}

template<class DataTypes>
void AspirationForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields() || (f_arrowSizeCoef.getValue()==0)) return;

    const VecCoord & x = *this->mstate->getX();
    RigidCoord position = f_positionRigid.getValue();

    Coord center = position.getCenter();
    Deriv force = f_force.getValue();
    Coord orientation = position.rotate(f_force.getValue());

    //double scale = f_scale.getValue();
    double ray = f_ray.getValue();

    simulation::getSimulation()->DrawUtility().drawArrow(center,center+(orientation*ray), f_arrowSizeCoef.getValue(), defaulttype::Vec<4,float>(0.1f,0.4f,1.0f,1.0f));

    glPushMatrix();
    glTranslatef(center.x(),center.y(),center.z());
    GLUquadric* params = gluNewQuadric();
    glColor3f(0.1,0.4,1.0);
    gluQuadricDrawStyle(params,GLU_LINE);
    gluSphere(params,ray,32,32);
    gluDeleteQuadric(params);
    glPopMatrix();


    for (unsigned int i=0; i<x.size(); i++)
    {
        Coord dir = x[i]-center;
        double norm = dir.norm();
        if (norm < ray)
        {
            Coord p1 = x[i];
            Coord p2 = x[i] - (dir*(1.0-(norm/ray)));

            if (dir*orientation>=0)
            {
                simulation::getSimulation()->DrawUtility().drawArrow(p1,p2, f_arrowSizeCoef.getValue(), defaulttype::Vec<4,float>(1.0f,0.4f,0.4f,1.0f));
            }
            else if (f_opposite_pressure.getValue()>0.0)
            {
                simulation::getSimulation()->DrawUtility().drawArrow(p1,p1-(orientation*f_opposite_pressure.getValue()), f_arrowSizeCoef.getValue(), defaulttype::Vec<4,float>(0.1f,1.0f,0.4f,1.0f));
            }
        }
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL



