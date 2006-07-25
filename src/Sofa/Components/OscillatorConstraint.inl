#ifndef SOFA_COMPONENTS_OscillatorConstraint_INL
#define SOFA_COMPONENTS_OscillatorConstraint_INL

#include "FixedConstraint.inl"
#include "OscillatorConstraint.h"
#include <math.h>

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes>
OscillatorConstraint<DataTypes>::OscillatorConstraint()
    : Core::Constraint<DataTypes>(NULL)
{
}


template <class DataTypes>
OscillatorConstraint<DataTypes>::OscillatorConstraint(Core::MechanicalModel<DataTypes>* mmodel)
    : Core::Constraint<DataTypes>(mmodel)
{
}

template <class DataTypes>
OscillatorConstraint<DataTypes>::~OscillatorConstraint()
{
}

template <class DataTypes>
OscillatorConstraint<DataTypes>*  OscillatorConstraint<DataTypes>::addConstraint(unsigned index, const Coord& mean, const Deriv& amplitude, Real pulsation, Real phase)
{
    this->constraints.push_back( std::make_pair( index, Oscillator(mean,amplitude,pulsation,phase) ) );
    return this;
}


template <class DataTypes>
void OscillatorConstraint<DataTypes>::projectResponse(VecDeriv& res)
{
    //Real t = (Real) getContext()->getTime();
    for( unsigned i=0; i<constraints.size(); ++i )
    {
        const unsigned& index = constraints[i].first;
        /*		const Deriv& a = constraints[i].second.amplitude;
        		const Real& w = constraints[i].second.pulsation;
        		const Real& p = constraints[i].second.phase;*/

        //res[index] = a*(-w)*w*sin(w*t+p);
        res[index] = Deriv();
    }
}

template <class DataTypes>
void OscillatorConstraint<DataTypes>::projectVelocity(VecDeriv& res)
{
    Real t = (Real) getContext()->getTime();
    for( unsigned i=0; i<constraints.size(); ++i )
    {
        const unsigned& index = constraints[i].first;
        const Deriv& a = constraints[i].second.amplitude;
        const Real& w = constraints[i].second.pulsation;
        const Real& p = constraints[i].second.phase;

        res[index] = a*w*cos(w*t+p);
    }
}

template <class DataTypes>
void OscillatorConstraint<DataTypes>::projectPosition(VecCoord& res)
{
    Real t = (Real) getContext()->getTime();
    //std::cerr<<"OscillatorConstraint<DataTypes>::projectPosition, t = "<<t<<endl;
    for( unsigned i=0; i<constraints.size(); ++i )
    {
        const unsigned& index = constraints[i].first;
        const Coord& m = constraints[i].second.mean;
        const Deriv& a = constraints[i].second.amplitude;
        const Real& w = constraints[i].second.pulsation;
        const Real& p = constraints[i].second.phase;

        res[index] = m + a*sin(w*t+p);
    }
}


//// Specialization for rigids
//template <>
//      void OscillatorConstraint<RigidTypes >::draw();
//template <>
//      void OscillatorConstraint<RigidTypes >::projectResponse(VecDeriv& dx);

} // namespace Components

} // namespace Sofa

#endif
