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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_CONSTRAINT_PARAMS_H
#define SOFA_CORE_CONSTRAINT_PARAMS_H

#include <sofa/core/ExecParams.h>
#include <sofa/core/MultiVecId.h>


namespace sofa
{

namespace core
{

/// Class gathering parameters use by constraint components methods, and transmitted by visitors
class ConstraintParams : public sofa::core::ExecParams
{
public:

    /// Description of the order of the constraint
    enum ConstOrder
    {
        POS = 0,
        VEL,
        ACC
    };

    /// @name Flags and parameters getters
    /// @{

    ConstOrder constOrder() const { return m_constOrder; }

    /// @}

    std::string getName() const
    {
        std::string result;
        switch ( m_constOrder )
        {
        case POS :
            result += "POSITION";
            break;
        case VEL :
            result += "VELOCITY";
            break;
        case ACC :
            result += "ACCELERATION";
            break;
        default :
            assert(false);
        }
        return result;
    }

    /// @name Access to vectors from a given state container (i.e. State or MechanicalState)
    /// @{

    /// Read access to current position vector
    template<class S>
    const Data<typename S::VecCoord>* readX(const S* state) const
    {   return m_x[state].read();    }

    /// Read access to current velocity vector
    template<class S>
    const Data<typename S::VecDeriv>* readV(const S* state) const
    {   return m_v[state].read();    }

    /// @}

    /// @name Setup methods
    /// Called by the OdeSolver from which the mechanical computations originate.
    /// They all return a reference to this MechanicalParam instance, to ease chaining multiple setup calls.

    /// @{

    const ConstMultiVecCoordId& x() const { return m_x; }
    ConstMultiVecCoordId& x()       { return m_x; }

    const ConstMultiVecDerivId& v() const { return m_v; }
    ConstMultiVecDerivId& v()       { return m_v; }

    /// Set the IDs of position vector
    ConstraintParams& setX(                   ConstVecCoordId v) { m_x.assign(v);   return *this; }
    ConstraintParams& setX(                   ConstMultiVecCoordId v) { m_x = v;   return *this; }
    template<class StateSet>
    ConstraintParams& setX(const StateSet& g, ConstVecCoordId v) { m_x.setId(g, v); return *this; }

    /// Set the IDs of velocity vector
    ConstraintParams& setV(                   ConstVecDerivId v) { m_v.assign(v);   return *this; }
    ConstraintParams& setV(                   ConstMultiVecDerivId v) { m_v = v;   return *this; }
    template<class StateSet>
    ConstraintParams& setV(const StateSet& g, ConstVecDerivId v) { m_v.setId(g, v); return *this; }

    /// @}


    /// Constructor, initializing all VecIds to default values, implicit and energy flags to false
    ConstraintParams(const sofa::core::ExecParams& p = sofa::core::ExecParams() )
        : sofa::core::ExecParams(p)
        , m_x (ConstVecCoordId::position())
        , m_v (ConstVecDerivId::velocity())
    {
    }

    /// Get the default MechanicalParams, to be used to provide a default values for method parameters
    static ConstraintParams* defaultInstance()
    {
        static ConstraintParams m_defaultInstance;

        return &m_defaultInstance;
    }

    ConstraintParams& setExecParams(const core::ExecParams* params)
    {
        sofa::core::ExecParams::operator=(*params);
        return *this;
    }

protected:
    /// Ids of position vector
    ConstMultiVecCoordId m_x;

    /// Ids of velocity vector
    ConstMultiVecDerivId m_v;

    /// Description of the order of the constraint
    ConstOrder m_constOrder;
};

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_CONSTRAINT_PARAMS_H
