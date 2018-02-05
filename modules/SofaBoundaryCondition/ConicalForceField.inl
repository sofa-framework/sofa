/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_INL

#include <SofaBoundaryCondition/ConicalForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Quat.h>

#include <sofa/helper/system/config.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void ConicalForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV )
{
    VecDeriv& f1 = *(dataF.beginEdit());
    const VecCoord& p1=dataX.getValue();
    const VecDeriv& v1=dataV.getValue();


    const Coord center = coneCenter.getValue();
    Real d = 0.0f;
    Real alpha = 0.0f;
    Real length_cp_prime = 0.0f;
    Coord p, p_prime, cp, dir, n_cp, cp_new, cp_prime, pp_prime, t;
    Coord n = coneHeight.getValue();
    n.normalize();
    Deriv force;

    this->contacts.beginEdit()->clear();
    f1.resize(p1.size());
    for (unsigned int i=0; i<p1.size(); i++)
    {
        p = p1[i];
        if (!isIn(p))
        {
            cp = p - center;
            //below the cone
            if (cp.norm() >  coneHeight.getValue().norm())
            {
                Real norm = cp.norm();
                d = norm - coneHeight.getValue().norm();
                dir = cp / norm;
                Real forceIntensity = -this->stiffness.getValue()*d;
                Real dampingIntensity = -this->damping.getValue()*d;
                force = cp*(forceIntensity/norm) - v1[i]*dampingIntensity;
                f1[i]+=force;
                Contact c;
                c.index = i;
                c.normal = cp / norm;
                c.pos = p;
                this->contacts.beginEdit()->push_back(c);

            }
            else
            {
                //side of the cone
                if (dot(cp,n) > 0)
                {
                    n_cp = cp/cp.norm();
                    alpha = (Real)( acos(n_cp*n) - coneAngle.getValue()*M_PI/180 );
                    t = n.cross(cp) ;
                    t /= t.norm();
                    defaulttype::Quat q(t, -alpha);
                    cp_new = q.rotate(cp);

                    cp_new.normalize();
                    length_cp_prime = dot(cp_new, cp);
                    cp_prime = cp_new * length_cp_prime;
                    p_prime = cp_prime + center;

                    pp_prime = p_prime - p;
                    d = pp_prime.norm();
                    dir = pp_prime/pp_prime.norm();
                    //sout << t << " " << alpha << sendl;
                }
                //top of the cone
                else
                {
                    d = cp.norm();
                    dir = (-cp)/cp.norm();
                }

                force = dir * (stiffness.getValue()*d);
                force += dir * (damping.getValue() * (-dot(dir,v1[i])));
                f1[i] += force;
                Contact c;
                c.index = i;
                c.normal = dir;
                c.pos = p;
                //c.fact = r / norm;
                this->contacts.beginEdit()->push_back(c);
            }
        }
    }
    this->contacts.endEdit();
    dataF.endEdit();
}

template<class DataTypes>
void ConicalForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& datadF , const DataVecDeriv& datadX)
{
    VecDeriv& df1 = *(datadF.beginEdit());
    const VecCoord& dx1=datadX.getValue();

    const Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df1.resize(dx1.size());
    for (unsigned int i=0; i<this->contacts.getValue().size(); i++)
    {
        const Contact& c = (*this->contacts.beginEdit())[i];
        assert((unsigned)c.index<dx1.size());
        Deriv du = dx1[c.index];
        Deriv dforce;
        dforce = (c.normal * ((du*c.normal)))*(-this->stiffness.getValue());
        df1[c.index] += dforce * kFact;
    }
    this->contacts.endEdit();

    datadF.endEdit();
}

template<class DataTypes>
void ConicalForceField<DataTypes>::updateStiffness( const VecCoord&  )
{
    serr<<"SphereForceField::updateStiffness-not-implemented !!!"<<sendl;
}

template<class DataTypes>
void ConicalForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!bDraw.getValue()) return;

    const Real a = coneAngle.getValue();
    Coord height = coneHeight.getValue();
    const Real h = sqrt(pow(coneHeight.getValue()[0],2) + pow(coneHeight.getValue()[1],2) +	pow(coneHeight.getValue()[2],2));
    const Real b = (Real)tan((a/180*M_PI)) * h;
    const Coord c = coneCenter.getValue();
//    Coord axis = height.cross(Coord(0,0,1));

    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND) ;
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) ;
    sofa::defaulttype::Vec4f color4(color.getValue()[0], color.getValue()[1], color.getValue()[2], 0.5);

    glPushMatrix();
    vparams->drawTool()->drawCone(c, c+height, 0, b, color4);
    glPopMatrix();

    glDisable(GL_BLEND) ;
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
#endif /* SOFA_NO_OPENGL */
}

template<class DataTypes>
bool ConicalForceField<DataTypes>::isIn(Coord p)
{
    const Coord c = coneCenter.getValue();
    const Coord height = coneHeight.getValue();
    const Real h = sqrt(pow(height[0],2) + pow(height[1],2) + pow(height[2],2));
    const Real distP = sqrt(pow(p[0] - c[0], 2) + pow(p[1] - c[1], 2) + pow(p[2] - c[2], 2));

    if (distP > h)
    {
        return false;
    }
    Coord vecP;

    for(unsigned i=0 ; i<3 ; ++i)
        vecP[i]=p[i]-c[i];

    if ( (acos(vecP*height/(h*distP))*180/M_PI) > coneAngle.getValue() )
    {

        return false;
    }
    return true;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_INL
