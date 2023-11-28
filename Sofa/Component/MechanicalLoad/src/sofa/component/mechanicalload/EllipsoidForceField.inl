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

#include <sofa/component/mechanicalload/EllipsoidForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/rmath.h>
#include <sofa/type/RGBAColor.h>
#include <cassert>
#include <iostream>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>


namespace sofa::component::mechanicalload
{

// v = sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1
// dv/dxj = xj/rj^2 * 1/sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)

// f  = -stiffness * v * (dv/dp) / norm(dv/dp)

// fi = -stiffness * (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1) * (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)

// dfi/dxj = -stiffness * [ d(sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)/dxj *   (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)     * d(xi/ri^2)/dxj / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)     *  (xi/ri^2) * d(1/sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4))/dxj ]
// dfi/dxj = -stiffness * [ xj/rj^2 * 1/sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2) * (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (i==j)/ri^2 / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (xi/ri^2) * (-1/2*2xj/rj^4*1/(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4) ]
// dfi/dxj = -stiffness * [ xj/rj^2 * 1/sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2) * (xi/ri^2) / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (i==j)/ri^2 / sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4)
//                          +  (sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2)-1)       * (xi/ri^2) * (-xj/rj^4*1/(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4) ]

// dfi/dxj = -stiffness * [ (xj/rj^2) * (xi/ri^2) * 1/(sqrt(x0^2/r0^2+x1^2/r1^2+x2^2/r2^2) * sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4))
//                          +  v       * (i==j) / (ri^2*sqrt(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4))
//                          +  v       * (xi/ri^2) * (xj/rj^2) * 1/(rj^2*(x0^2/r0^4+x1^2/r1^4+x2^2/r2^4) ]

template<class DataTypes>
EllipsoidForceField<DataTypes>::EllipsoidForceField()
    : d_contacts(initData(&d_contacts,"contacts", "Vector of contacts"))
    , d_center(initData(&d_center, "center", "ellipsoid center"))
    , d_vradius(initData(&d_vradius, "vradius", "ellipsoid radius"))
    , d_stiffness(initData(&d_stiffness, (Real)500, "stiffness", "force stiffness (positive to repulse outward, negative inward)"))
    , d_damping(initData(&d_damping, (Real)5, "damping", "force damping"))
    , d_color(initData(&d_color, sofa::type::RGBAColor(0.0f,0.5f,1.0f,1.0f), "color", "ellipsoid color. (default=0,0.5,1.0,1.0)"))
{
}

template <class DataTypes>
EllipsoidForceField<DataTypes>::~EllipsoidForceField()
{
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::setStiffness(Real stiff)
{
    d_stiffness.setValue( stiff );
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::setDamping(Real damp)
{
    d_damping.setValue( damp );
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV )
{

    VecDeriv& f1        = *(dataF.beginEdit());
    const VecCoord& p1  =   dataX.getValue()  ;
    const VecDeriv& v1  =   dataV.getValue()  ;


    const Coord center = this->d_center.getValue();
    const Coord r = this->d_vradius.getValue();
    const Real stiffness = this->d_stiffness.getValue();
    const Real stiffabs = helper::rabs(stiffness);
    //const Real s2 = (stiff < 0 ? - stiff*stiff : stiff*stiff );
    Coord inv_r2;
    for (unsigned int j=0; j<N; j++)
    {
        inv_r2[j] = 1 / (r[j] * r[j]);
    }
    sofa::type::vector<Contact>* contacts = this->d_contacts.beginEdit();
    contacts->clear();
    f1.resize(p1.size());
    for (Size i=0; i<p1.size(); i++)
    {
        Coord dp = p1[i] - center;
        Real norm2 = 0;
        for (unsigned int j=0; j<N; j++) norm2 += (dp[j]*dp[j])*inv_r2[j];
        //Real d = (norm2-1)*s2;
        if ((norm2-1)*stiffness<0)
        {
            Real norm = helper::rsqrt(norm2);
            Real v = norm-1;
            Coord grad;
            for (unsigned int j=0; j<N; j++)
            {
                grad[j] = dp[j]*inv_r2[j];
            }
            Real gnorm2 = grad.norm2();
            Real gnorm = helper::rsqrt(gnorm2);
            //grad /= gnorm; //.normalize();
            Real forceIntensity = -stiffabs*v/gnorm;
            Real dampingIntensity = this->d_damping.getValue()*helper::rabs(v);
            Deriv force = grad*forceIntensity - v1[i]*dampingIntensity;
            f1[i]+=force;
            Contact c;
            c.index = i;
            Real fact1 = -stiffabs / (norm * gnorm);
            Real fact2 = -stiffabs*v / gnorm;
            Real fact3 = -stiffabs*v / gnorm2;
            for (unsigned int ci = 0; ci < N; ++ci)
            {
                for (unsigned int cj = 0; cj < N; ++cj)
                    c.m[ci][cj] = grad[ci]*grad[cj] * (fact1 + fact3*inv_r2[cj]);
                c.m[ci][ci] += fact2*inv_r2[ci];
            }
            contacts->push_back(c);
        }
    }
    this->d_contacts.endEdit();

    dataF.endEdit();
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX )
{
    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());
    helper::WriteAccessor df1 = sofa::helper::getWriteAccessor(datadF);
    const VecCoord& dx1 = datadX.getValue();

    df1.resize(dx1.size());
    for (const Contact& c : this->d_contacts.getValue())
    {
        assert((unsigned)c.index < dx1.size());
        const Deriv du = dx1[c.index];
        Deriv dforce = c.m * du;
        dforce *= kFactor;
        df1[c.index] += dforce;
    }
}

template <class DataTypes>
void EllipsoidForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const Contact& contact : this->d_contacts.getValue())
    {
        const auto locationInGlobalMatrix = contact.index * Deriv::total_size;
        dfdx(locationInGlobalMatrix, locationInGlobalMatrix) += contact.m;
    }
}

template <class DataTypes>
void EllipsoidForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
SReal EllipsoidForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

template<class DataTypes>
void EllipsoidForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    Real cx=0, cy=0, cz=0;
    DataTypes::get(cx, cy, cz, d_center.getValue());
    Real rx=1, ry=1, rz=1;
    DataTypes::get(rx, ry, rz, d_vradius.getValue());
    const sofa::type::Vec3 radii(rx, ry, (d_stiffness.getValue()>0 ? rz : -rz));
    const sofa::type::Vec3 vCenter(cx, cy, cz);

    vparams->drawTool()->enableLighting();
    
    vparams->drawTool()->setMaterial(d_color.getValue());
	vparams->drawTool()->drawEllipsoid(vCenter, radii);
    vparams->drawTool()->disableLighting();



}

} // namespace sofa::component::mechanicalload
