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

namespace sofa::core
{

/// Class gathering parameters use by mechanical components methods, and transmitted by mechanical visitors
class SOFA_CORE_API MechanicalParams : public sofa::core::ExecParams
{
public:

    /// @name Flags and parameters getters
    /// @{

    /// Time step
    SReal dt() const { return m_dt; }

    /// Is the time integration scheme implicit ?
    bool implicit() const { return m_implicit; }



    /// Mass matrix contributions factor (for implicit schemes)
    SReal mFactor() const { return m_mFactor; }

    /// Damping matrix contributions factor (for implicit schemes)
    SReal bFactor() const { return m_bFactor; }

    /// Stiffness matrix contributions factor (for implicit schemes)
    SReal kFactor() const { setKFactorUsed(true); return m_kFactor; }


    /** @name Rayleigh Damping D = rayleighStiffness*K - rayleighMass*M
     */
    /// @{

    /// \returns kfactor +  bfactor*rayleighStiffness
    SReal kFactorIncludingRayleighDamping( SReal rayleighStiffness ) const { return kFactor() + bFactor()*rayleighStiffness; }
    /// \returns mfactor +  bfactor*rayleighMass
    SReal mFactorIncludingRayleighDamping( SReal rayleighMass ) const { return mFactor() - bFactor()*rayleighMass; }

    /// @}

    SOFA_ATTRIBUTE_DEPRECATED__SYMMETRICMATRIX("Use supportOnlySymmetricMatrix instead.")
    bool symmetricMatrix() const { return supportOnlySymmetricMatrix(); }

    /// Symmetric matrix flag, for solvers specialized on symmetric matrices
    bool supportOnlySymmetricMatrix() const { return m_supportOnlySymmetricMatrix; }

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

    /// @name Access to vectors from a given SingleLink to a state container (i.e. State or MechanicalState)
    /// @{

    /// Read access to current position vector
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readX(state.get())")
    const Data<typename S::VecCoord>* readX(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_x[state.get()].read();    }

    /// Read access to current velocity vector
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readV(state.get())")
    const Data<typename S::VecDeriv>* readV(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_v[state.get()].read();    }

    /// Read access to current force vector
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readF(state.get())")
    const Data<typename S::VecDeriv>* readF(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_f[state.get()].read();    }

    /// Read access to current dx vector (for implicit schemes)
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readDx(state.get())")
    const Data<typename S::VecDeriv>* readDx(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_dx[state.get()].read();    }

    /// Read access to current df vector (for implicit schemes)
    template<class Owner, class S, unsigned int flags>
    SOFA_ATTRIBUTE_DISABLED__READX_PARAMS("To fix your code use readDf(state.get())")
    const Data<typename S::VecDeriv>* readDf(const sofa::core::objectmodel::SingleLink<Owner,S,flags>& state) const
    {   return m_df[state.get()].read();    }

    /// @}

    /// @name Setup methods
    /// Called by the OdeSolver from which the mechanical computations originate.
    /// They all return a reference to this MechanicalParam instance, to ease chaining multiple setup calls.

    /// @{

    /// Set time step
    MechanicalParams& setDt(SReal v) { m_dt = v; return *this; }

    /// Specify if the time integration scheme is implicit
    MechanicalParams& setImplicit(bool v) { m_implicit = v; return *this; }

    /// Set Mass matrix contributions factor (for implicit schemes)
    MechanicalParams& setMFactor(SReal v) { m_mFactor = v; return *this; }

    /// Set Damping matrix contributions factor (for implicit schemes)
    MechanicalParams& setBFactor(SReal v) { m_bFactor = v; return *this; }

    /// Set Stiffness matrix contributions factor (for implicit schemes)
    MechanicalParams& setKFactor(SReal v) { m_kFactor = v; return *this; }

    SOFA_ATTRIBUTE_DEPRECATED__SYMMETRICMATRIX("Use setSupportOnlySymmetricMatrix instead.")
    MechanicalParams& setSymmetricMatrix(bool b) { return setSupportOnlySymmetricMatrix(b); }

    /// Set the flag (for implicit schemes) specifying if solvers are only specialized for symmetric matrices
    MechanicalParams& setSupportOnlySymmetricMatrix(bool b) { m_supportOnlySymmetricMatrix = b; return *this; }

#ifndef NDEBUG
    /// Checks wether or nor kFactor is used in ForceFields. Temporary here for compatiblity reasons
    void setKFactorUsed(bool b) const { m_kFactorUsed = b; }
    bool getKFactorUsed() const { return m_kFactorUsed; }
protected:
    /// Checks if the stiffness matrix contributions factor has been accessed
    mutable bool m_kFactorUsed;
public:
#else
    void setKFactorUsed(bool) const {}
#endif

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
    MechanicalParams(const sofa::core::ExecParams& p = sofa::core::ExecParams() );

    /// Copy Constructor
    MechanicalParams( const MechanicalParams& mparams );

    /// Get the default MechanicalParams, to be used to provide a default values for method parameters
    static const MechanicalParams* defaultInstance();

    MechanicalParams* setExecParams(const core::ExecParams* params);

    MechanicalParams* operator= ( const MechanicalParams& mparams );


protected:

    /// Time step
    SReal m_dt;

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
    SReal m_mFactor;

    /// Damping matrix contributions factor (for implicit schemes)
    SReal m_bFactor;

    /// Stiffness matrix contributions factor (for implicit schemes)
    SReal m_kFactor;

    /// True if a symmetric matrix is assumed in the left-hand term of the dynamics equations, for solvers specialized on symmetric matrices
    bool m_supportOnlySymmetricMatrix;

    /// @name Experimental compliance API
    /// @{
    SReal m_implicitVelocity;  ///< ratio of future and current force used for velocity update    (1 is fully implicit, 0 is fully explicit)
    SReal m_implicitPosition;  ///< ratio of future and current velocity used for position update (1 is fully implicit, 0 is fully explicit)

public:
    void setImplicitVelocity( SReal i ) { m_implicitVelocity = i; }
    const SReal& implicitVelocity() const { return m_implicitVelocity; }
    void setImplicitPosition( SReal i ) { m_implicitPosition = i; }
    const SReal& implicitPosition() const { return m_implicitPosition; }
    /// @}
};

} // namespace sofa::core
