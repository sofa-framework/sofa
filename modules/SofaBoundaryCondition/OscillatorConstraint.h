/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/**
 * Apply sinusoidal trajectories to particles.
 * Defined as \f$ x = x_m A \sin ( \omega t + \phi )\f$
 * where \f$ x_m, A , \omega t , \phi \f$ are the mean value, the amplitude, the pulsation and the phase, respectively.
 */
template <class TDataTypes>
class OscillatorConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OscillatorConstraint,TDataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

protected:
    struct Oscillator
    {
        unsigned int index;
        Coord mean;
        Deriv amplitude;
        Real pulsation;
        Real phase;

        Oscillator()
        {
        }

        Oscillator(unsigned int i, const Coord& m, const Deriv& a,
                const Real& w, const Real& p) :
            index(i), mean(m), amplitude(a), pulsation(w), phase(p)
        {
        }

        inline friend std::istream& operator >>(std::istream& in, Oscillator& o)
        {
            in >> o.index >> o.mean >> o.amplitude >> o.pulsation >> o.phase;
            return in;
        }

        inline friend std::ostream& operator <<(std::ostream& out, const Oscillator& o)
        {
            out << o.index << " " << o.mean << " " << o.amplitude << " "
                << o.pulsation << " " << o.phase << "\n";
            return out;
        }
    };

    Data< helper::vector< Oscillator > > constraints; ///< constrained particles


public:
    OscillatorConstraint();

    OscillatorConstraint(core::behavior::MechanicalState<TDataTypes>* mstate);

    ~OscillatorConstraint();
public:
    OscillatorConstraint<TDataTypes>* addConstraint(unsigned index, const Coord& mean, const Deriv& amplitude, Real pulsation, Real phase);

    // -- Constraint interface


    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData);
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData);
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData);
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData);

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class OscillatorConstraint<defaulttype::Rigid3dTypes>;
extern template class OscillatorConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class OscillatorConstraint<defaulttype::Rigid3fTypes>;
extern template class OscillatorConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
