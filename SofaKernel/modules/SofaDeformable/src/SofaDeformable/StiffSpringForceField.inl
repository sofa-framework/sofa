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
#include <sofa/linearalgebra/BaseMatrix.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::interactionforcefield
{

template<class DataTypes>
StiffSpringForceField<DataTypes>::StiffSpringForceField(double ks, double kd)
    : StiffSpringForceField<DataTypes>(nullptr, ks, kd)
{
}

template<class DataTypes>
StiffSpringForceField<DataTypes>::StiffSpringForceField(MechanicalState* mstate, double ks, double kd)
    : SpringForceField<DataTypes>(mstate, ks, kd)
    , d_indices1(initData(&d_indices1, "indices1", "Indices of the source points on the first model"))
    , d_indices2(initData(&d_indices2, "indices2", "Indices of the fixed points on the second model"))
    , d_length(initData(&d_length, 0.0, "length", "uniform length of all springs"))
{
    this->addUpdateCallback("updateSprings", { &d_indices1, &d_indices2, &d_length, &this->ks, &this->kd}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        createSpringsFromInputs();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {&this->springs});
}


template<class DataTypes>
void StiffSpringForceField<DataTypes>::init()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);

    if (d_indices1.isSet() && d_indices2.isSet())
    {
        createSpringsFromInputs();
    }
    this->SpringForceField<DataTypes>::init();

    if (this->getMState())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class DataTypes>
void StiffSpringForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x,
    const DataVecDeriv& v)
{
    dfdx.resize(this->springs.getValue().size());
    Inherit1::addForce(mparams, f, x, v);
}

template <class DataTypes>
void StiffSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df,
    const DataVecDeriv& dx)
{
    VecDeriv& _df = *sofa::helper::getWriteAccessor(df);
    const VecDeriv& _dx = *sofa::helper::getReadAccessor(dx);

    const Real kFactor = static_cast<Real>(sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(
        mparams, this->rayleighStiffness.getValue()));
    const Real bFactor = static_cast<Real>(sofa::core::mechanicalparams::bFactor(mparams));

    const type::vector<Spring>& springs = this->springs.getValue();
    _df.resize(_dx.size());

    for (sofa::Index i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(_df,_dx,_df,_dx, i, springs[i], kFactor, bFactor);
    }
}


template<class DataTypes>
void StiffSpringForceField<DataTypes>::createSpringsFromInputs()
{
    if (d_indices1.getValue().size() != d_indices2.getValue().size())
    {
        msg_error() << "Inputs indices sets sizes are different: d_indices1: " << d_indices1.getValue().size()
            << " | d_indices2 " << d_indices2.getValue().size()
            << " . No springs will be created";
        return;
    }

    msg_info() << "Inputs have changed, recompute  Springs From Data Inputs";

    type::vector<Spring>& _springs = *this->springs.beginEdit();
    _springs.clear();

    const SetIndexArray & indices1 = d_indices1.getValue();
    const SetIndexArray & indices2 = d_indices2.getValue();

    const SReal& _ks = this->ks.getValue();
    const SReal& _kd = this->kd.getValue();
    const SReal& _length = d_length.getValue();
    for (sofa::Index i = 0; i<indices1.size(); ++i)
        _springs.push_back(Spring(indices1[i], indices2[i], _ks, _kd, _length));

    this->springs.endEdit();
}


template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringForce(
        Real& potentialEnergy,
        VecDeriv& f1,
        const  VecCoord& p1,
        const VecDeriv& v1,
        VecDeriv& f2,
        const  VecCoord& p2,
        const  VecDeriv& v2,
        sofa::Index i,
        const Spring& spring)
{
    //    this->cpt_addForce++;
    sofa::Index a = spring.m1;
    sofa::Index b = spring.m2;

    /// Get the positional part out of the dofs.
    typename DataTypes::CPos u = DataTypes::getCPos(p2[b])-DataTypes::getCPos(p1[a]);
    Real d = u.norm();
    if( spring.enabled && d>1.0e-9 && (!spring.elongationOnly || d>spring.initpos))
    {
        // F =   k_s.(l-l_0 ).U + k_d((V_b - V_a).U).U = f.U   where f is the intensity and U the direction
        Real inverseLength = 1.0f/d;
        u *= inverseLength;
        Real elongation = (Real)(d - spring.initpos);
        potentialEnergy += elongation * elongation * spring.ks / 2;
        typename DataTypes::DPos relativeVelocity = DataTypes::getDPos(v2[b])-DataTypes::getDPos(v1[a]);
        Real elongationVelocity = dot(u,relativeVelocity);
        Real forceIntensity = (Real)(spring.ks*elongation+spring.kd*elongationVelocity);
        typename DataTypes::DPos force = u*forceIntensity;

        DataTypes::setDPos( f1[a], DataTypes::getDPos(f1[a]) + force ) ;
        DataTypes::setDPos( f2[b], DataTypes::getDPos(f2[b]) - force ) ;

        // Compute stiffness dF/dX
        // The force change dF comes from length change dl and unit vector change dU:
        // dF = k_s.dl.U + f.dU
        // dU = 1/l.(I-U.U^T).dX   where dX = dX_1 - dX_0  and I is the identity matrix
        // dl = U^T.dX
        // dF = k_s.U.U^T.dX + f/l.(I-U.U^T).dX = ((k_s-f/l).U.U^T + f/l.I).dX
        Mat& m = this->dfdx[i];
        Real tgt = forceIntensity * inverseLength;
        for(sofa::Index j=0; j<N; ++j )
        {
            for(sofa::Index k=0; k<N; ++k )
            {
                m[j][k] = ((Real)spring.ks-tgt) * u[j] * u[k];
            }
            m[j][j] += tgt;
        }
    }
    else // null length, no force and no stiffness
    {
        Mat& m = this->dfdx[i];
        for(sofa::Index j=0; j<N; ++j )
        {
            for(sofa::Index k=0; k<N; ++k )
            {
                m[j][k] = 0;
            }
        }
    }
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, sofa::Index i, const Spring& spring, double kFactor, double /*bFactor*/)
{
    const sofa::Index a = spring.m1;
    const sofa::Index b = spring.m2;
    const typename DataTypes::CPos d = DataTypes::getDPos(dx2[b]) - DataTypes::getDPos(dx1[a]);
    typename DataTypes::DPos dforce = this->dfdx[i]*d;

    dforce *= kFactor;

    DataTypes::setDPos( df1[a], DataTypes::getDPos(df1[a]) + dforce ) ;
    DataTypes::setDPos( df2[b], DataTypes::getDPos(df2[b]) - dforce ) ;
}

template<class DataTypes>
void StiffSpringForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    if (!this->mstate)
        return;

    const Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,this->rayleighStiffness.getValue());

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat = matrix->getMatrix(this->getMState());
    if (!mat) return;
    const sofa::type::vector<Spring >& ss = this->springs.getValue();
    const sofa::Size n = ss.size() < this->dfdx.size() ? sofa::Size(ss.size()) : sofa::Size(this->dfdx.size());
    for (sofa::Index e = 0; e < n; ++e)
    {
        const Spring& s = ss[e];
        const sofa::Index p1 = mat.offset + Deriv::total_size * s.m1;
        const sofa::Index p2 = mat.offset + Deriv::total_size * s.m2;
        const Mat& m = this->dfdx[e];
        for(sofa::Index i=0; i<N; i++)
        {
            for (sofa::Index j=0; j<N; j++)
            {
                Real k = (Real)(m[i][j]*kFact);
                mat.matrix->add(p1+i,p1+j, -k);
                mat.matrix->add(p1+i,p2+j,  k);
                mat.matrix->add(p2+i,p1+j,  k);//or mat->add(p1+j,p2+i, k);
                mat.matrix->add(p2+i,p2+j, -k);
            }
        }
    }
}

} // namespace sofa::component::interactionforcefield
