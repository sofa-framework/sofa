#ifndef SOFA_COMPONENTS_OscillatorConstraint_H
#define SOFA_COMPONENTS_OscillatorConstraint_H

#include "Sofa/Core/Constraint.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include <Sofa/Components/Common/vector.h>


namespace Sofa
{

namespace Components
{

/** Apply sinusoidal trajectories. Defined as \f$ x = x_m A \sin ( \omega t + \phi )\f$
where \f$ x_m, A , \omega t , \phi \f$ are the mean value, the amplitude, the pulsation and the phase, respectively.
*/
template <class DataTypes>
class OscillatorConstraint : public Core::Constraint<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:
    struct Oscillator
    {
        Coord mean;
        Deriv amplitude;
        Real pulsation;
        Real phase;
        Oscillator( const Coord& m, const Deriv& a, const Real& w, const Real& p )
            : mean(m), amplitude(a), pulsation(w), phase(p) {}
    };
    Common::vector< std::pair<unsigned,Oscillator> > constraints; ///< constrained particles


public:
    OscillatorConstraint();

    OscillatorConstraint(Core::MechanicalModel<DataTypes>* mmodel);

    ~OscillatorConstraint();

    OscillatorConstraint<DataTypes>* addConstraint(unsigned index, const Coord& mean, const Deriv& amplitude, Real pulsation, Real phase);

    // -- Constraint interface
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/); ///< project x to constrained space (x models a position)

    // -- Visual interface
    void draw() {}

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
