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
#include <sofa/core/ExecParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/ConstraintOrder.h>

namespace sofa::core
{

/// Class gathering parameters use by constraint components methods, and transmitted by visitors
/// read the velocity and position
/// and where the
class SOFA_CORE_API ConstraintParams : public sofa::core::ExecParams
{
public:

    SOFA_ATTRIBUTE_DEPRECATED__CONSTORDER() static constexpr auto POS = sofa::core::ConstraintOrder::POS;
    SOFA_ATTRIBUTE_DEPRECATED__CONSTORDER() static constexpr auto VEL = sofa::core::ConstraintOrder::VEL;
    SOFA_ATTRIBUTE_DEPRECATED__CONSTORDER() static constexpr auto ACC = sofa::core::ConstraintOrder::ACC;
    SOFA_ATTRIBUTE_DEPRECATED__CONSTORDER() static constexpr auto POS_AND_VEL = sofa::core::ConstraintOrder::POS_AND_VEL;

    /// @name Flags and parameters getters
    /// @{

    ConstraintOrder constOrder() const { return m_constOrder; }

    ConstraintParams& setOrder(ConstraintOrder o) { m_constOrder = o;   return *this; }

	/// Smooth contribution factor (for smooth constraints resolution)
    double smoothFactor() const { return m_smoothFactor; }

    /// @}

    [[nodiscard]] std::string_view getName() const
    {
        return constOrderToString(m_constOrder);
    }

    /// @name Access to vectors from a given state container (i.e. State or MechanicalState)
    /// @{

    /// Read access to the free (unconstrained) position vector
    template<class S>
    const Data<typename S::VecCoord>* readX(const S* state) const
    {   return m_x[state].read();    }

    /// Read access to the free (unconstrained) velocity vector
    template<class S>
    const Data<typename S::VecDeriv>* readV(const S* state) const
    {   return m_v[state].read();    }

    /// Read access to the constraint jacobian matrix
    template<class S>
    const Data<typename S::MatrixDeriv>* readJ(const S* state) const
    {
        return m_j[state].read();
    }

    /// Read access to the constraint force vector
    template<class S>
    const Data<typename S::VecDeriv>* readLambda(S* state) const
    {
        return m_lambda[state].read();
    }

    /// Read access to the constraint corrective motion vector
    template<class S>
    const Data<typename S::VecDeriv>* readDx(S* state) const
    {
        return m_dx[state].read();
    }




    /// @name Access to vectors from a given SingleLink to a state container (i.e. State or MechanicalState)
    /// @{

    /// Read access to the free (unconstrained) position
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readX(state.get())")
    const Data<typename S::VecCoord>* readX(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_x[state.get()].read();    }

    /// Read access to the free (unconstrained) velocity vector
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readV(state.get())")
    const Data<typename S::VecDeriv>* readV(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_v[state.get()].read();    }

    /// Read access to the constraint jacobian matrix
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readJ(state.get())")
    const Data<typename S::MatrixDeriv>* readJ(const sofa::core::objectmodel::SingleLink<Owner, S, flags>& state) const
    {
        return m_j[state.get()].read();
    }

    /// Read access to the constraint force vector
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readLambda(state.get())")
    const Data<typename S::VecDeriv>* readLambda(sofa::core::objectmodel::SingleLink<Owner, S, flags>& state) const
    {
        return m_lambda[state.get(this)].read();
    }

    /// Read access to the constraint corrective motion vector
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readDx(state.get())")
    const Data<typename S::VecDeriv>* readDx(sofa::core::objectmodel::SingleLink<Owner, S, flags>& state) const
    {
        return m_dx[state.get(this)].read();
    }


    /// @}

    /// @name Setup methods
    /// Called by the OdeSolver from which the mechanical computations originate.
    /// They all return a reference to this MechanicalParam instance, to ease chaining multiple setup calls.

    /// @{

	/// Set smooth contribution factor (for smooth constraints resolution)
    ConstraintParams& setSmoothFactor(double v) { m_smoothFactor = v; return *this; }

    /// Returns ids of the position vectors
    const ConstMultiVecCoordId& x() const { return m_x; }
    /// Returns ids of the position vectors
    ConstMultiVecCoordId& x()       { return m_x; }

    /// Returns ids of the velocity vectors
    const ConstMultiVecDerivId& v() const { return m_v; }
    /// Returns ids of the velocity vectors
    ConstMultiVecDerivId& v()       { return m_v; }

    /// Returns ids of the constraint jacobian matrices
    const MultiMatrixDerivId&  j() const { return m_j; }
    /// Returns ids of the constraint jacobian matrices
    MultiMatrixDerivId& j()              { return m_j; }

    /// Returns ids of the contraint correction vectors
    const MultiVecDerivId& dx() const { return m_dx;  }
    /// Returns ids of the contraint correction vectors
    MultiVecDerivId&  dx()            { return m_dx;  }

    /// Returns ids of the constraint lambda vectors
    const MultiVecDerivId& lambda() const { return m_lambda; }
    /// Returns ids of the constraint lambda vectors
    MultiVecDerivId&  lambda()            { return m_lambda; }

    /// Set the IDs where to read the free position vector
    ConstraintParams& setX(                   ConstVecCoordId v) { m_x.assign(v);   return *this; }
    ConstraintParams& setX(                   ConstMultiVecCoordId v) { m_x = v;   return *this; }
    template<class StateSet>
    ConstraintParams& setX(const StateSet& g, ConstVecCoordId v) { m_x.setId(g, v); return *this; }

    /// Set the IDs where to read the free velocity vector
    ConstraintParams& setV(                   ConstVecDerivId v) { m_v.assign(v);   return *this; }
    ConstraintParams& setV(                   ConstMultiVecDerivId v) { m_v = v;   return *this; }
    template<class StateSet>
    ConstraintParams& setV(const StateSet& g, ConstVecDerivId v) { m_v.setId(g, v); return *this; }

    /// Set the IDs where to read the constraint jacobian matrix
    ConstraintParams& setJ(MatrixDerivId j)      { m_j.assign(j); return *this; }
    ConstraintParams& setJ(MultiMatrixDerivId j) { m_j = j; return *this; }
    template<class StateSet>
    ConstraintParams& setJ(const StateSet& g, MatrixDerivId j) { m_j.setId(g, j); return *this; }

    /// Set the IDs where to write corrective displacement vector
    ConstraintParams& setDx(VecDerivId dx)      { m_dx.assign(dx); return *this; }
    ConstraintParams& setDx(MultiVecDerivId dx) { m_dx = dx;   return *this; }
    template<class StateSet>
    ConstraintParams& setDx(const StateSet& g, MultiVecDerivId dx) { m_dx.setId(g, dx); return *this; }

    /// Set the IDs where to write the constraint force vector
    ConstraintParams& setLambda(VecDerivId lambda) { m_lambda.assign(lambda); return *this; }
    ConstraintParams& setLambda(MultiVecDerivId lambda) { m_lambda = lambda;   return *this; }
    template<class StateSet>
    ConstraintParams& setLambda(const StateSet& g, MultiVecDerivId lambda) { m_lambda.setId(g, lambda); return *this; }

    /// @}


    /// Constructor, initializing all VecIds to default values, implicit and energy flags to false
    ConstraintParams(const sofa::core::ExecParams& p = *sofa::core::execparams::defaultInstance());

    /// Get the default MechanicalParams, to be used to provide a default values for method parameters
    static const ConstraintParams* defaultInstance();

    ConstraintParams& setExecParams(const core::ExecParams* params);

protected:
    /// Ids of position vector
    ConstMultiVecCoordId m_x;

    /// Ids of velocity vector
    ConstMultiVecDerivId m_v;

    /// Ids of the constraint jacobian matrix
    MultiMatrixDerivId m_j;

    /// Ids of contraint correction vector
    MultiVecDerivId      m_dx;

    /// Ids of constraint lambda vector
    MultiVecDerivId      m_lambda;

    /// Description of the order of the constraint
    ConstraintOrder m_constOrder;

	/// Smooth contribution factor (for smooth constraints resolution)
    double m_smoothFactor;
};

} // namespace sofa::core
