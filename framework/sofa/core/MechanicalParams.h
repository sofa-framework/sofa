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
#ifndef SOFA_CORE_MECHANICAL_PARAMS_H
#define SOFA_CORE_MECHANICAL_PARAMS_H

#include <sofa/core/ExecParams.h>
#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace core
{

/// Class gathering parameters use by mechanical components methods, and transmitted by mechanical visitors
class MechanicalParams : public sofa::core::ExecParams
{
public:

    enum StateGroup
    {
        SGROUP_DEFAULT = 0,
        SGROUP_ACTIVE  = 1,
    };

    /// Time step
    double dt() const { return m_dt; }

    /// Is the time integration scheme implicit ?
    bool implicit() const { return m_implicit; }

    /// Get the "group" of a given State, depending on whether it is controlled by the current solver
    template<class StateContainer>
    StateGroup getGroup(const StateContainer* state) const
    {
        bool isActive = (m_solverContext && state->getContext()->hasAncestor(m_solverContext));
        return isActive ? SGROUP_ACTIVE:SGROUP_DEFAULT;
    }

    /// Read access to current position vector
    template<class S>
    const typename S::VecCoord* getX(const S* state) const
    {   return state->getVCoord(m_x[getGroup(state)]);    }
    /// Write access to current position vector
    template<class S>
    typename S::VecCoord* getX(S* state) const
    {   return state->getVCoord(m_x[getGroup(state)]);    }

    /// Read access to current velocity vector
    template<class S>
    const typename S::VecDeriv* getV(const S* state) const
    {   return state->getVDeriv(m_v[getGroup(state)]);    }
    /// Write access to current velocity vector
    template<class S>
    typename S::VecDeriv* getV(S* state) const
    {   return state->getVDeriv(m_v[getGroup(state)]);    }

    /// Read access to current force vector
    template<class S>
    const typename S::VecDeriv* getF(const S* state) const
    {   return state->getVDeriv(m_f[getGroup(state)]);    }
    /// Write access to current force vector
    template<class S>
    typename S::VecDeriv* getF(S* state) const
    {   return state->getVDeriv(m_f[getGroup(state)]);    }

    /// Read access to current dx vector (for implicit schemes)
    template<class S>
    const typename S::VecDeriv* getDx(const S* state) const
    {   return state->getVDeriv(m_dx[getGroup(state)]);    }
    /// Write access to current dx vector (for implicit schemes)
    template<class S>
    typename S::VecDeriv* getDx(S* state) const
    {   return state->getVDeriv(m_dx[getGroup(state)]);    }

    /// Read access to current df vector (for implicit schemes)
    template<class S>
    const typename S::VecDeriv* getDf(const S* state) const
    {
        return state->getVDeriv(m_df[getGroup(state)]);
    }
    /// Write access to current dx vector (for implicit schemes)
    template<class S>
    typename S::VecDeriv* getDf(S* state) const
    {   return state->getVDeriv(m_df[getGroup(state)]);    }

    /// Mass matrix contributions factor (for implicit schemes)
    double mFactor() const { return m_mFactor; }

    /// Damping matrix contributions factor (for implicit schemes)
    double bFactor() const { return m_bFactor; }

    /// Stiffness matrix contributions factor (for implicit schemes)
    double kFactor() const { return m_kFactor; }

    MechanicalParams(const sofa::core::ExecParams& p = sofa::core::ExecParams() )
        : sofa::core::ExecParams(p)
        , m_dt(0.0)
        , m_implicit(true)
        , m_solverContext(NULL)
        , m_x(VecId::position(), VecId::position())
        , m_v(VecId::velocity(), VecId::velocity())
        , m_f(VecId::force(), VecId::force())
        , m_dx(VecId::dx(), VecId::dx())
        , m_df(VecId::dforce(), VecId::dforce())
        , m_mFactor(0)
        , m_bFactor(0)
        , m_kFactor(0)
    {
    }

    /// Set time step
    MechanicalParams& setDt(double v) { m_dt = v; return *this; }

    /// Specify if the time integration scheme implicit
    MechanicalParams& setImplicit(bool v) { m_implicit = v; return *this; }

    /// Set the context of the current solver, used to determine which states are "active"
    MechanicalParams& setSolverContext(core::objectmodel::BaseContext* v) { m_solverContext = v; return *this; }

    /// Set the IDs of position vector
    MechanicalParams& setX(VecId v)               { m_x.assign(v); return *this; }
    MechanicalParams& setX(VecId v, StateGroup g) { m_x[g]  =  v ; return *this; }

    /// Set the IDs of velocity vector
    MechanicalParams& setV(VecId v)               { m_v.assign(v); return *this; }
    MechanicalParams& setV(VecId v, StateGroup g) { m_v[g]  =  v ; return *this; }

    /// Set the IDs of force vector
    MechanicalParams& setF(VecId v)               { m_f.assign(v); return *this; }
    MechanicalParams& setF(VecId v, StateGroup g) { m_f[g]  =  v ; return *this; }

    /// Set the IDs of dx vector (for implicit schemes)
    MechanicalParams& setDx(VecId v)               { m_dx.assign(v); return *this; }
    MechanicalParams& setDx(VecId v, StateGroup g) { m_dx[g]  =  v ; return *this; }

    /// Set the IDs of df vector (for implicit schemes)
    MechanicalParams& setDf(VecId v)               { m_df.assign(v); return *this; }
    MechanicalParams& setDf(VecId v, StateGroup g) { m_df[g]  =  v ; return *this; }

    /// Set Mass matrix contributions factor (for implicit schemes)
    MechanicalParams& setMFactor(double v) { m_mFactor = v; return *this; }

    /// Set Damping matrix contributions factor (for implicit schemes)
    MechanicalParams& setBFactor(double v) { m_bFactor = v; return *this; }

    /// Set Stiffness matrix contributions factor (for implicit schemes)
    MechanicalParams& setKFactor(double v) { m_kFactor = v; return *this; }

protected:

    enum { NGROUPS = 2 };

    /// Time step
    double m_dt;

    /// Is the time integration scheme implicit ?
    bool m_implicit;

    /// Context of the current solver, used to determine which states are "active"
    core::objectmodel::BaseContext* m_solverContext;

    /// Ids of position vector
    helper::fixed_array<VecId,NGROUPS> m_x;

    /// Ids of velocity vector
    helper::fixed_array<VecId,NGROUPS> m_v;

    /// Ids of force vector
    helper::fixed_array<VecId,NGROUPS> m_f;

    /// Ids of dx vector (for implicit schemes)
    helper::fixed_array<VecId,NGROUPS> m_dx;

    /// Ids of df vector (for implicit schemes)
    helper::fixed_array<VecId,NGROUPS> m_df;

    /// Mass matrix contributions factor (for implicit schemes)
    double m_mFactor;

    /// Damping matrix contributions factor (for implicit schemes)
    double m_bFactor;

    /// Stiffness matrix contributions factor (for implicit schemes)
    double m_kFactor;

};

} // namespace core

} // namespace sofa

#endif
