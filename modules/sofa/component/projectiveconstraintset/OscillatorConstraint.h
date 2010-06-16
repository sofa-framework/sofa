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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_H

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/** Apply sinusoidal trajectories to particles. Defined as \f$ x = x_m A \sin ( \omega t + \phi )\f$
	where \f$ x_m, A , \omega t , \phi \f$ are the mean value, the amplitude, the pulsation and the phase, respectively.
	*/
template <class DataTypes>
class OscillatorConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OscillatorConstraint,DataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:
    struct Oscillator
    {
        unsigned int index;
        Coord mean;
        Deriv amplitude;
        Real pulsation;
        Real phase;

        Oscillator() {}

        Oscillator( unsigned int i, const Coord& m, const Deriv& a, const Real& w, const Real& p )
            : index(i), mean(m), amplitude(a), pulsation(w), phase(p) {}

        inline friend std::istream& operator >> ( std::istream& in, Oscillator& o )
        {
            in>>o.index>>o.mean>>o.amplitude>>o.pulsation>>o.phase;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Oscillator& o )
        {
            out << o.index<< " " <<o.mean<< " " <<o.amplitude<< " " <<o.pulsation<< " " <<o.phase<<"\n";
            return out;
        }
    };

    Data< helper::vector< Oscillator > > constraints; ///< constrained particles


public:
    OscillatorConstraint();

    OscillatorConstraint(core::behavior::MechanicalState<DataTypes>* mstate);

    ~OscillatorConstraint();

    OscillatorConstraint<DataTypes>* addConstraint(unsigned index, const Coord& mean, const Deriv& amplitude, Real pulsation, Real phase);

    // -- Constraint interface
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx);

    void projectResponse(VecDeriv& dx);
    void projectResponse(SparseVecDeriv& dx);

    virtual void projectVelocity(VecDeriv& /*dx*/); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/); ///< project x to constrained space (x models a position)

    void draw() {}

    /// this constraint is holonomic
    bool isHolonomic() {return true;}
};

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
