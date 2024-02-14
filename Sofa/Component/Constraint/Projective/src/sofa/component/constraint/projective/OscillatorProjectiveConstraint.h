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
#include <sofa/component/constraint/projective/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/type/vector.h>


namespace sofa::component::constraint::projective
{

/**
 * Apply sinusoidal trajectories to particles.
 * Defined as \f$ x = x_m A \sin ( \omega t + \phi )\f$
 * where \f$ x_m, A , \omega t , \phi \f$ are the mean value, the amplitude, the pulsation and the phase, respectively.
 */
template <class TDataTypes>
class OscillatorProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OscillatorProjectiveConstraint,TDataTypes),SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,TDataTypes));

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

        Oscillator();
        Oscillator(unsigned int i, const Coord& m, const Deriv& a,
                   const Real& w, const Real& p);

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

    Data< type::vector< Oscillator > > constraints; ///< constrained particles


public:
    explicit OscillatorProjectiveConstraint(core::behavior::MechanicalState<TDataTypes>* mstate=nullptr);
    ~OscillatorProjectiveConstraint() override ;

    OscillatorProjectiveConstraint<TDataTypes>* addConstraint(unsigned index,
                                                    const Coord& mean, const Deriv& amplitude,
                                                    Real pulsation, Real phase);

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

protected:
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx,
        const std::function<void(DataDeriv&, const unsigned int)>& clear);
};


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_OSCILLATORPROJECTIVECONSTRAINT_CPP)
extern template class OscillatorProjectiveConstraint<defaulttype::Rigid3Types>;
extern template class OscillatorProjectiveConstraint<defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::constraint::projective
