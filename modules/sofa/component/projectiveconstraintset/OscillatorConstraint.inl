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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_INL

#include <sofa/component/projectiveconstraintset/FixedConstraint.inl>
#include <sofa/component/projectiveconstraintset/OscillatorConstraint.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;

template <class TDataTypes>
OscillatorConstraint<TDataTypes>::OscillatorConstraint()
    : core::behavior::ProjectiveConstraintSet<TDataTypes>(NULL)
    , constraints(initData(&constraints,"oscillators","Define a sequence of oscillating particules: \n[index, mean, amplitude, pulsation, phase]"))
{
}


template <class TDataTypes>
OscillatorConstraint<TDataTypes>::OscillatorConstraint(core::behavior::MechanicalState<TDataTypes>* mstate)
    : core::behavior::ProjectiveConstraintSet<TDataTypes>(mstate)
    , constraints(initData(&constraints,"oscillators","Define a sequence of oscillating particules: \n[index, Mean(x,y,z), amplitude(x,y,z), pulsation, phase]"))
{
}

template <class TDataTypes>
OscillatorConstraint<TDataTypes>::~OscillatorConstraint()
{
}

template <class TDataTypes>
OscillatorConstraint<TDataTypes>*  OscillatorConstraint<TDataTypes>::addConstraint(unsigned index, const Coord& mean, const Deriv& amplitude, Real pulsation, Real phase)
{
    this->constraints.beginEdit()->push_back( Oscillator(index,mean,amplitude,pulsation,phase) );
    return this;
}


template <class TDataTypes> template <class DataDeriv>
void OscillatorConstraint<TDataTypes>::projectResponseT(DataDeriv& res, const core::MechanicalParams* /*mparams*/)
{
    const helper::vector<Oscillator> &oscillators = constraints.getValue();
    //Real t = (Real) this->getContext()->getTime();
    for (unsigned i = 0; i < oscillators.size(); ++i)
    {
        const unsigned& index = oscillators[i].index;
        //const Deriv& a = constraints[i].second.amplitude;
        //const Real& w = constraints[i].second.pulsation;
        //const Real& p = constraints[i].second.phase;

        //res[index] = a*(-w)*w*sin(w*t+p);
        res[index] = Deriv();
    }
}

template <class TDataTypes>
void OscillatorConstraint<TDataTypes>::projectResponse(DataVecDeriv& resData, const core::MechanicalParams* mparams)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT(res.wref(), mparams);
}

template <class TDataTypes>
void OscillatorConstraint<TDataTypes>::projectVelocity(DataVecDeriv& vData, const core::MechanicalParams* /*mparams*/)
{
    helper::WriteAccessor<DataVecDeriv> v = vData;
    const helper::vector<Oscillator>& oscillators = constraints.getValue();
    Real t = (Real) this->getContext()->getTime();
    for (unsigned i = 0; i < oscillators.size(); ++i)
    {
        const unsigned& index = oscillators[i].index;
        const Deriv& a = oscillators[i].amplitude;
        const Real& w = oscillators[i].pulsation;
        const Real& p = oscillators[i].phase;

        v[index] = a * w * cos(w * t + p);
    }
}

template <class TDataTypes>
void OscillatorConstraint<TDataTypes>::projectPosition(DataVecCoord& xData, const core::MechanicalParams* /*mparams*/)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    const helper::vector<Oscillator> &oscillators = constraints.getValue();
    Real t = (Real) this->getContext()->getTime();
    for (unsigned i = 0; i < oscillators.size(); ++i)
    {
        const unsigned& index = oscillators[i].index;
        const Coord& m = oscillators[i].mean;
        const Deriv& a = oscillators[i].amplitude;
        const Real& w = oscillators[i].pulsation;
        const Real& p = oscillators[i].phase;

        x[index] = m + a * sin(w * t + p);
    }
}

template <class TDataTypes>
void OscillatorConstraint<TDataTypes>::projectJacobianMatrix(DataMatrixDeriv& cData, const core::MechanicalParams* mparams)
{
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseT<MatrixDerivRowType>(rowIt.row(), mparams);
        ++rowIt;
    }
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
