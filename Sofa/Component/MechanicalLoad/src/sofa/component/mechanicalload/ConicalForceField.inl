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

#include <sofa/component/mechanicalload/ConicalForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/Quat.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/helper/rmath.h>
#include <sofa/core/MechanicalParams.h>
#include <cassert>
#include <iostream>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>


namespace sofa::component::mechanicalload
{

template<class DataTypes>
ConicalForceField<DataTypes>::ConicalForceField()
    : coneCenter(initData(&coneCenter, "coneCenter", "cone center"))
    , coneHeight(initData(&coneHeight, "coneHeight", "cone height"))
    , coneAngle(initData(&coneAngle, (Real)10, "coneAngle", "cone angle"))

    , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness"))
    , damping(initData(&damping, (Real)5, "damping", "force damping"))
    , color(initData(&color, sofa::type::RGBAColor::blue(), "color", "cone color. (default=0.0,0.0,0.0,1.0,1.0)"))
{
}

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
                    type::Quat<SReal> q(t, -alpha);
                    cp_new = q.rotate(cp);

                    cp_new.normalize();
                    length_cp_prime = dot(cp_new, cp);
                    cp_prime = cp_new * length_cp_prime;
                    p_prime = cp_prime + center;

                    pp_prime = p_prime - p;
                    d = pp_prime.norm();
                    dir = pp_prime/pp_prime.norm();
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

    const Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

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

template <class DataTypes>
void ConicalForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    const auto k = stiffness.getValue();

    for (const auto& contact : contacts.getValue())
    {
        const auto localMatrix = -k * sofa::type::dyad(contact.normal, contact.normal);
        dfdx(contact.index * Deriv::total_size, contact.index * Deriv::total_size)
            += localMatrix;
    }

}

template <class DataTypes>
void ConicalForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void ConicalForceField<DataTypes>::updateStiffness( const VecCoord&  )
{
    msg_error() << "SphereForceField::updateStiffness-not-implemented !!!";
}

template<class DataTypes>
void ConicalForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;

    const Real a = coneAngle.getValue();
    Coord height = coneHeight.getValue();
    const Real h = sqrt(pow(coneHeight.getValue()[0],2) + pow(coneHeight.getValue()[1],2) +	pow(coneHeight.getValue()[2],2));
    const Real b = (Real)tan((a/180*M_PI)) * h;
    const Coord c = coneCenter.getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    vparams->drawTool()->enableBlending();
    vparams->drawTool()->enableLighting();

    sofa::type::RGBAColor rgbcolor(color.getValue()[0], color.getValue()[1], color.getValue()[2], 0.5);

    vparams->drawTool()->drawCone(c, c+height, 0, b, rgbcolor);
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

template<class DataTypes>
void ConicalForceField<DataTypes>::setCone(const Coord& center, Coord height, Real angle)
{
    coneCenter.setValue( center );
    coneHeight.setValue( height );
    coneAngle.setValue( angle );
}

template<class DataTypes>
void ConicalForceField<DataTypes>::setStiffness(Real stiff)
{
    stiffness.setValue( stiff );
}

template<class DataTypes>
void ConicalForceField<DataTypes>::setDamping(Real damp)
{
    damping.setValue( damp );
}

} // namespace sofa::component::mechanicalload
