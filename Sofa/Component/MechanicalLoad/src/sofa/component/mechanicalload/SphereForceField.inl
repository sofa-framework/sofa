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

#include <sofa/component/mechanicalload/SphereForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/rmath.h>
#include <cassert>
#include <iostream>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/linearalgebra/BaseMatrix.h>

namespace sofa::component::mechanicalload
{

// f  = -stiffness * (x -c ) * (|x-c|-r)/|x-c|
// fi = -stiffness * (xi-ci) * (|x-c|-r)/|x-c|
// dfi/dxj = -stiffness * ( (xi-ci) * d((|x-c|-r)/|x-c|)/dxj + d(xi-ci)/dxj * (|x-c|-r)/|x-c| )
// d(xi-ci)/dxj = 1 if i==j, 0 otherwise
// d((|x-c|-r)/|x-c|)/dxj = (|x-c|*d(|x-c|-r)/dxj - d(|x-c|)/dxj * (|x-c|-r))/|x-c|^2
//                        = (d(|x-c|)/dxj * (|x-c| - |x-c| + r))/|x-c|^2
//                        = r/|x-c|^2 * d(|x-c|)/dxj
//                        = r/|x-c|^2 * d(sqrt(sum((xi-ci)^2)))/dxj
//                        = r/|x-c|^2 * 1/2 * 1/sqrt(sum((xi-ci)^2)) * d(sum(xi-ci)^2)/dxj
//                        = r/|x-c|^2 * 1/2 * 1/|x-c| * d((xj-cj)^2)/dxj
//                        = r/|x-c|^2 * 1/2 * 1/|x-c| * (2(xj-cj))
//                        = r/|x-c|^2 * (xj-cj)/|x-c|
// dfi/dxj = -stiffness * ( (xi-ci) * r/|x-c|^2 * (xj-cj)/|x-c| + (i==j) * (|x-c|-r)/|x-c| )
//         = -stiffness * ( (xi-ci)/|x-c| * (xj-cj)/|x-c| * r/|x-c| + (i==j) * (1 - r/|x-c|) )
// df = -stiffness * ( (x-c)/|x-c| * dot(dx,(x-c)/|x-c|) * r/|x-c|   + dx * (1 - r/|x-c|) )

template<class DataTypes>
SphereForceField<DataTypes>::SphereForceField()
    : contacts(initData(&contacts,"contacts", "Contacts"))
    , sphereCenter(initData(&sphereCenter, "center", "sphere center"))
    , sphereRadius(initData(&sphereRadius, (Real)1, "radius", "sphere radius"))
    , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness"))
    , damping(initData(&damping, (Real)5, "damping", "force damping"))
    , color(initData(&color, sofa::type::RGBAColor(0.0f,0.0f,1.0f, 1.0f), "color", "sphere color. (default=[0,0,1,1])"))
    , localRange( initData(&localRange, type::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
    , bilateral( initData(&bilateral, false, "bilateral", "if true the sphere force field is applied on both sides"))
{
}

template<class DataTypes>
void SphereForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f1 = *d_f.beginEdit();
    const VecCoord& p1 = d_x.getValue();
    const VecDeriv& v1 = d_v.getValue();

    const Coord center = sphereCenter.getValue();
    const Real r = sphereRadius.getValue();
    const Real r2 = r*r;
    this->contacts.beginEdit()->clear();
    f1.resize(p1.size());

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Coord dp = p1[i] - center;
        Real norm2 = dp.norm2();
        if (norm2<r2 || bilateral.getValue() )
        {
            Real norm = helper::rsqrt(norm2);
            Real d = norm - r;
            Real forceIntensity = -this->stiffness.getValue()*d;
            Real dampingIntensity = -this->damping.getValue()*d;
            Deriv force = dp*(forceIntensity/norm) - v1[i]*dampingIntensity;
            f1[i]+=force;
            Contact c;
            c.index = i;
            c.normal = dp / norm;
            c.fact = r / norm;
            this->contacts.beginEdit()->push_back(c);
        }
    }
    this->contacts.endEdit();
    d_f.endEdit();
}

template<class DataTypes>
void SphereForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal kFactor, unsigned int &offset)
{
    const Real fact = (Real)(-this->stiffness.getValue()*kFactor);
    for (unsigned int i=0; i<this->contacts.getValue().size(); i++)
    {
        const Contact& c = (this->contacts.getValue())[i];
        unsigned int p = c.index;
        for (sofa::Index l=0; l<Deriv::total_size; ++l)
            for (sofa::Index k=0; k<Deriv::total_size; ++k)
            {
                SReal coef = (c.normal[l] * c.normal[k] * c.fact + (l==k ? (1 - c.fact) : (SReal)0.0)) * fact;
                mat->add(offset + p*Deriv::total_size + l, offset + p*Deriv::total_size + k, coef);
            }
    }
}

template<class DataTypes>
void SphereForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df1 = *d_df.beginEdit();
    const VecDeriv& dx1 = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    df1.resize(dx1.size());
    const Real fact = (Real)(-this->stiffness.getValue()*kFactor);
    for (unsigned int i=0; i<this->contacts.getValue().size(); i++)
    {
        const Contact& c = (this->contacts.getValue())[i];
        assert((unsigned)c.index<dx1.size());
        Deriv du = dx1[c.index];
        Deriv dforce; dforce = (c.normal * ((du*c.normal)*c.fact) + du * (1 - c.fact)) * fact;
        df1[c.index] += dforce;
    }

    d_df.endEdit();
}

template<class DataTypes>
void SphereForceField<DataTypes>::updateStiffness( const VecCoord& x )
{
    const Coord center = sphereCenter.getValue();
    const Real r = sphereRadius.getValue();
    const Real r2 = r*r;
    this->contacts.beginEdit()->clear();

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Coord dp = x[i] - center;
        Real norm2 = dp.norm2();
        if (norm2<r2)
        {
            Real norm = helper::rsqrt(norm2);
            Contact c;
            c.index = i;
            c.normal = dp / norm;
            c.fact = r / norm;
            this->contacts.beginEdit()->push_back(c);
        }
    }
    this->contacts.endEdit();
}

template <class DataTypes>
void SphereForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    const Real fact = (Real)(-this->stiffness.getValue());

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto& c : sofa::helper::getReadAccessor(contacts))
    {
        unsigned int p = c.index;
        for (sofa::Index l = 0; l < Deriv::total_size; ++l)
        {
            for (sofa::Index k = 0; k < Deriv::total_size; ++k)
            {
                const SReal coef = (c.normal[l] * c.normal[k] * c.fact + (l == k ? (1 - c.fact) : (SReal)0.0)) * fact;
                dfdx(p * Deriv::total_size + l, p * Deriv::total_size + k) += coef;
            }
        }
    }
}

template <class DataTypes>
void SphereForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void SphereForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    type::Vec3d center;
    DataTypes::get(center[0], center[1], center[2], sphereCenter.getValue());
    const Real& r = sphereRadius.getValue();

    vparams->drawTool()->enableLighting();

    vparams->drawTool()->drawSphere(center, (float)(r*0.99) );
    vparams->drawTool()->disableLighting();


}

template<class DataTypes>
SReal SphereForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

template<class DataTypes>
void SphereForceField<DataTypes>::setSphere(const Coord& center, Real radius)
{
    sphereCenter.setValue( center );
    sphereRadius.setValue( radius );
}

template<class DataTypes>
void SphereForceField<DataTypes>::setStiffness(Real stiff)
{
    stiffness.setValue( stiff );
}

template<class DataTypes>
void SphereForceField<DataTypes>::setDamping(Real damp)
{
    damping.setValue( damp );
}

} // namespace sofa::component::mechanicalload
