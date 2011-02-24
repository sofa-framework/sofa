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
#include <sofa/core/MultiVecId.h>

namespace sofa
{

namespace core
{

/// Class gathering parameters use by mechanical components methods, and transmitted by mechanical visitors
class MechanicalParams : public sofa::core::ExecParams
{
public:

    /// @name Flags and parameters getters
    /// @{

    /// Time step
    double dt() const { return m_dt; }

    /// Is the time integration scheme implicit ?
    bool implicit() const { return m_implicit; }

    /// Mass matrix contributions factor (for implicit schemes)
    double mFactor() const { return m_mFactor; }

    /// Damping matrix contributions factor (for implicit schemes)
    double bFactor() const { return m_bFactor; }

    /// Stiffness matrix contributions factor (for implicit schemes)
    double kFactor() const { setKFactorUsed(true); return m_kFactor; }

    /// Symmetric matrix flag, for solvers specialized on symmetric matrices
    bool symmetricMatrix() const { return m_symmetricMatrix; }

    /// Should the kinematic and potential energies be computed ?
    bool energy() const { return m_energy; }

    /// @}

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

    /// Read access to current force vector
    template<class S>
    const Data<typename S::VecDeriv>* readF(const S* state) const
    {   return m_f[state].read();    }

    /// Read access to current dx vector (for implicit schemes)
    template<class S>
    const Data<typename S::VecDeriv>* readDx(const S* state) const
    {   return m_dx[state].read();    }

    /// Read access to current df vector (for implicit schemes)
    template<class S>
    const Data<typename S::VecDeriv>* readDf(const S* state) const
    {   return m_df[state].read();    }

    /// @}

    /// @name Setup methods
    /// Called by the OdeSolver from which the mechanical computations originate.
    /// They all return a reference to this MechanicalParam instance, to ease chaining multiple setup calls.

    /// @{

    /// Set time step
    MechanicalParams& setDt(double v) { m_dt = v; return *this; }

    /// Specify if the time integration scheme is implicit
    MechanicalParams& setImplicit(bool v) { m_implicit = v; return *this; }

    /// Set Mass matrix contributions factor (for implicit schemes)
    MechanicalParams& setMFactor(double v) { m_mFactor = v; return *this; }

    /// Set Damping matrix contributions factor (for implicit schemes)
    MechanicalParams& setBFactor(double v) { m_bFactor = v; return *this; }

    /// Set Stiffness matrix contributions factor (for implicit schemes)
    MechanicalParams& setKFactor(double v) { m_kFactor = v; return *this; }

    /// Set the symmetric matrix flag (for implicit schemes), for solvers specialized on symmetric matrices
    MechanicalParams& setSymmetricMatrix(bool b) { m_symmetricMatrix = b; return *this; }

    /// Checks wether or nor kFactor is used in ForceFields. Temporary here for compatiblity reasons
    void setKFactorUsed(bool b) const { m_kFactorUsed = b; }
    bool getKFactorUsed() const { return m_kFactorUsed; }

    /// Specify if the potential and kinematic energies should be computed ?
    MechanicalParams& setEnergy(bool v) { m_energy = v; return *this; }

    const ConstMultiVecCoordId& x() const { return m_x; }
    ConstMultiVecCoordId& x()       { return m_x; }

    const ConstMultiVecDerivId& v() const { return m_v; }
    ConstMultiVecDerivId& v()       { return m_v; }

    const ConstMultiVecDerivId& f() const { return m_f; }
    ConstMultiVecDerivId& f()       { return m_f; }

    const ConstMultiVecDerivId& dx() const { return m_dx; }
    ConstMultiVecDerivId& dx()       { return m_dx; }

    const ConstMultiVecDerivId& df() const { return m_df; }
    ConstMultiVecDerivId& df()       { return m_df; }

    /// Set the IDs of position vector
    MechanicalParams& setX(                   ConstVecCoordId v) { m_x.assign(v);   return *this; }
    MechanicalParams& setX(                   ConstMultiVecCoordId v) { m_x = v;   return *this; }
    template<class StateSet>
    MechanicalParams& setX(const StateSet& g, ConstVecCoordId v) { m_x.setId(g, v); return *this; }

    /// Set the IDs of velocity vector
    MechanicalParams& setV(                   ConstVecDerivId v) { m_v.assign(v);   return *this; }
    MechanicalParams& setV(                   ConstMultiVecDerivId v) { m_v = v;   return *this; }
    template<class StateSet>
    MechanicalParams& setV(const StateSet& g, ConstVecDerivId v) { m_v.setId(g, v); return *this; }

    /// Set the IDs of force vector
    MechanicalParams& setF(                   ConstVecDerivId v) { m_f.assign(v);   return *this; }
    MechanicalParams& setF(                   ConstMultiVecDerivId v) { m_f = v;   return *this; }
    template<class StateSet>
    MechanicalParams& setF(const StateSet& g, ConstVecDerivId v) { m_f.setId(g, v); return *this; }

    /// Set the IDs of dx vector (for implicit schemes)
    MechanicalParams& setDx(                   ConstVecDerivId v) { m_dx.assign(v);   return *this; }
    MechanicalParams& setDx(                   ConstMultiVecDerivId v) { m_dx = v;   return *this; }
    template<class StateSet>
    MechanicalParams& setDx(const StateSet& g, ConstVecDerivId v) { m_dx.setId(g, v); return *this; }

    /// Set the IDs of df vector (for implicit schemes)
    MechanicalParams& setDf(                   ConstVecDerivId v) { m_df.assign(v);   return *this; }
    MechanicalParams& setDf(                   ConstMultiVecDerivId v) { m_df = v;   return *this; }
    template<class StateSet>
    MechanicalParams& setDf(const StateSet& g, ConstVecDerivId v) { m_df.setId(g, v); return *this; }

    /// @}

    /// Constructor, initializing all VecIds to default values, implicit and energy flags to false
    MechanicalParams(const sofa::core::ExecParams& p = sofa::core::ExecParams() )
        : sofa::core::ExecParams(p)
        , m_dt(0.0)
        , m_implicit(false)
        , m_energy(false)
        , m_x (ConstVecCoordId::position())
        , m_v (ConstVecDerivId::velocity())
        , m_f (ConstVecDerivId::force())
        , m_dx(ConstVecDerivId::dx())
        , m_df(ConstVecDerivId::dforce())
        , m_mFactor(0)
        , m_bFactor(0)
        , m_kFactor(0)
        , m_symmetricMatrix(true)
    {
    }

    /// Get the default MechanicalParams, to be used to provide a default values for method parameters
    static MechanicalParams* defaultInstance()
    {
        static MechanicalParams m_defaultInstance;

        return &m_defaultInstance;
    }

    MechanicalParams* setExecParams(const core::ExecParams* params)
    {
        sofa::core::ExecParams::operator=(*params);
        return this;
    }

protected:

    /// Time step
    double m_dt;

    /// Is the time integration scheme implicit ?
    bool m_implicit;

    /// Should the kinematic and potential energies be computed ?
    bool m_energy;

    /// Ids of position vector
    ConstMultiVecCoordId m_x;

    /// Ids of velocity vector
    ConstMultiVecDerivId m_v;

    /// Ids of force vector
    ConstMultiVecDerivId m_f;

    /// Ids of dx vector (for implicit schemes)
    ConstMultiVecDerivId m_dx;

    /// Ids of df vector (for implicit schemes)
    ConstMultiVecDerivId m_df;

    /// Mass matrix contributions factor (for implicit schemes)
    double m_mFactor;

    /// Damping matrix contributions factor (for implicit schemes)
    double m_bFactor;

    /// Stiffness matrix contributions factor (for implicit schemes)
    double m_kFactor;

    /// Checks if the stiffness matrix contributions factor has been accessed
    mutable bool m_kFactorUsed;

    /// True if a symmetric matrix is assumed in the left-hand term of the dynamics equations, for solvers specialized on symmetric matrices
    double m_symmetricMatrix;
};

} // namespace core

} // namespace sofa

#endif
