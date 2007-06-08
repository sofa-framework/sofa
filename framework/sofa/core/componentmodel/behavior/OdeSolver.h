/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_ODESOLVER_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_ODESOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief Component responsible for timestep integration, i.e. advancing the state from time t to t+dt.
 *
 *  This class currently control both the integration scheme (explicit,
 *  implicit, static, etc), and the linear system resolution algorithm
 *  (conjugate gradient, matrix direct inversion, etc). Those two aspect will
 *  propably be separated in a future version.
 *
 *  While all computations required to do the integration step are handled by
 *  this object, they should not be implemented directly in it, but instead
 *  the solver propagates orders (or Action) to the other components in the
 *  scenegraph that will locally execute them. This allow for greater
 *  flexibility (the solver can just ask for the forces to be computed without
 *  knowing what type of forces are present), as well as performances
 *  (some computations can be executed in parallel).
 *
 */
class OdeSolver : public objectmodel::BaseObject
{
public:
    typedef BaseMechanicalState::VecId VecId;

    OdeSolver();

    virtual ~OdeSolver();

    /// Main computation method.
    ///
    /// Specify and execute all computation for timestep integration, i.e.
    /// advancing the state from time t to t+dt.
    virtual void solve (double dt) = 0;

    /// Method called at initialization, during the backwards traversal of the data structure.
    virtual void bwdInit() {}

    /// @name Actions and MultiVectors
    /// These methods provides an abstract view of the mechanical system to animate.
    /// They are implemented by executing Actions in the subtree of the scene-graph below this solver.
    /// @{

    /// @name Vector operations
    /// Most of these operations can be hidden by using the MultiVector class.
    /// @{

    /// Wait for the completion of previous operations and return the result of the last v_dot call.
    ///
    /// Note that currently all methods are blocking so finish simply return the result of the last v_dot call.
    virtual double finish();

    /// Allocate a temporary vector
    virtual VecId v_alloc(VecId::Type t);
    /// Free a previously allocated temporary vector
    virtual void v_free(VecId v);

    virtual void v_clear(VecId v); ///< v=0
    virtual void v_eq(VecId v, VecId a); ///< v=a
    virtual void v_peq(VecId v, VecId a, double f=1.0); ///< v+=f*a
    virtual void v_teq(VecId v, double f); ///< v*=f
    virtual void v_dot(VecId a, VecId b); ///< a dot b ( get result using finish )
    /// Propagate the given displacement through all mappings
    virtual void propagateDx(VecId dx);
    /// Apply projective constraints to the given vector
    virtual void projectResponse(VecId dx, double **W=NULL);
    virtual void addMdx(VecId res, VecId dx); ///< res += M.dx
    virtual void integrateVelocity(VecId res, VecId x, VecId v, double dt); ///< res = x + v.dt
    virtual void accFromF(VecId a, VecId f); ///< a = M^-1 . f
    /// Propagate the given state (time, position and velocity) through all mappings
    virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    /// Compute the current force (given the latest propagated position and velocity)
    virtual void computeForce(VecId result);
    /// Compute the current force delta (given the latest propagated displacement)
    virtual void computeDf(VecId df);
    /// Compute the acceleration corresponding to the given state (time, position and velocity)
    virtual void computeAcc(double t, VecId a, VecId x, VecId v);

    virtual void computeContactForce(VecId result);
    virtual void computeContactDf(VecId df);
    virtual void computeContactAcc(double t, VecId a, VecId x, VecId v);

    /// @}

    /// @name Matrix operations
    /// @{

    // BaseMatrix & BaseVector Computations

    virtual void addMBK_ToMatrix(defaulttype::BaseMatrix *A, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0);
    virtual void addMBKdx_ToVector(VecId res, VecId dx, double mFact=1.0, double bFact=1.0, double kFact=1.0);
    virtual void getMatrixDimension(unsigned int * const, unsigned int * const);
    virtual void multiVector2BasicVector(VecId src, defaulttype::BaseVector *dest=NULL, unsigned int offset=0);

//    virtual void computeMatrix(defaulttype::SofaBaseMatrix *mat=NULL, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0);
//    virtual void computeOpVector(defaulttype::SofaBaseVector *vect=NULL, unsigned int offset=0);
//    virtual void matResUpdatePosition(defaulttype::SofaBaseVector *vect=NULL, unsigned int offset=0);

    /// @}

    /// @name Debug operations
    /// @{

    /// Dump the content of the given vector.
    virtual void print( VecId v, std::ostream& out );
    virtual void printWithElapsedTime( VecId v,  unsigned time, std::ostream& out=std::cerr );

    /// @}

    /// @}

protected:
    //defaulttype::SofaBaseMatrix *mat;

    /// Helper class allocating IDs to temporary vectors.
    class VectorIndexAlloc
    {
    protected:
        std::set<unsigned int> vused; ///< Currently in-use vectors
        std::set<unsigned int> vfree; ///< Once used vectors
        unsigned int  maxIndex; ///< Max index used
    public:
        VectorIndexAlloc();
        /// Retrieve a unused ID
        unsigned int alloc();
        /// Free a previously retrieved ID
        bool free(unsigned int v);
    };
    std::map<VecId::Type, VectorIndexAlloc > vectors; ///< Current temporary vectors

    /// Result of latest v_dot operation
    double result;

    /// Helper class providing a high-level view of underlying state vectors.
    ///
    /// It is used to convert math-like operations to call to computation methods.
    class MultiVector
    {
    public:
        typedef OdeSolver::VecId VecId;

    protected:
        /// Solver who is using this vector
        core::componentmodel::behavior::OdeSolver* parent;

        /// Identifier of this vector
        VecId v;

        /// Copy-constructor is forbidden
        MultiVector(const MultiVector& v);

    public:
        /// Refers to a state vector with the given ID (VecId::position(), VecId::velocity(), etc).
        MultiVector(core::componentmodel::behavior::OdeSolver* parent, VecId v) : parent(parent), v(v)
        {}

        /// Allocate a new temporary vector with the given type (VecId::V_COORD or VecId::V_DERIV).
        MultiVector(core::componentmodel::behavior::OdeSolver* parent, VecId::Type t) : parent(parent), v(parent->v_alloc(t))
        {}

        ~MultiVector()
        {
            parent->v_free(v);
        }

        /// Automatic conversion to the underlying VecId
        operator VecId()
        {
            return v;
        }

        /// v = 0
        void clear()
        {
            parent->v_clear(v);
        }

        /// v = a
        void eq(VecId a)
        {
            parent->v_eq(v, a);
        }

        /// v += a*f
        void peq(VecId a, double f=1.0)
        {
            parent->v_peq(v, a, f);
        }
        /// v *= f
        void teq(double f)
        {
            parent->v_teq(v, f);
        }
        /// \return v.a
        double dot(VecId a)
        {
            parent->v_dot(v, a);
            return parent->finish();
        }

        /// \return sqrt(v.v)
        double norm()
        {
            parent->v_dot(v, v);
            return sqrt( parent->finish() );
        }

        /// v = a
        void operator=(VecId a)
        {
            eq(a);
        }

        /// v = a
        void operator=(const MultiVector& a)
        {
            eq(a.v);
        }

        /// v += a
        void operator+=(VecId a)
        {
            peq(a);
        }

        /// v -= a
        void operator-=(VecId a)
        {
            peq(a,-1);
        }

        /// v *= f
        void operator*=(double f)
        {
            teq(f);
        }

        /// v /= f
        void operator/=(double f)
        {
            teq(1.0/f);
        }

        /// return the scalar product dot(v,a)
        double operator*(VecId a)
        {
            return dot(a);
        }

        friend std::ostream& operator << (std::ostream& out, const MultiVector& mv )
        {
            mv.parent->print(mv.v,out);
            return out;
        }
    };
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
