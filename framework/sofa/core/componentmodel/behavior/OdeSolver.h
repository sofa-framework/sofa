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
 *  the solver propagates orders (or Visitor) to the other components in the
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

    /// @name Visitors and MultiVectors
    /// These methods provides an abstract view of the mechanical system to animate.
    /// They are implemented by executing Visitors in the subtree of the scene-graph below this solver.
    /// @{

    /// @name Vector operations
    /// Most of these operations can be hidden by using the MultiVector class.
    /// @{

    /// Wait for the completion of previous operations and return the result of the last v_dot call.
    ///
    /// Note that currently all methods are blocking so finish simply return the result of the last v_dot call.
    virtual double finish() = 0;

    /// Allocate a temporary vector
    virtual VecId v_alloc(VecId::Type t) = 0;
    /// Free a previously allocated temporary vector
    virtual void v_free(VecId v) = 0;

    virtual void v_clear(VecId v) = 0; ///< v=0
    virtual void v_eq(VecId v, VecId a) = 0; ///< v=a
    virtual void v_peq(VecId v, VecId a, double f=1.0) = 0; ///< v+=f*a
    virtual void v_teq(VecId v, double f) = 0; ///< v*=f
    virtual void v_dot(VecId a, VecId b) = 0; ///< a dot b ( get result using finish )
    virtual void v_threshold(VecId a, double threshold) = 0; ///< nullify the values below the given threshold
    /// Propagate the given displacement through all mappings
    virtual void propagateDx(VecId dx) = 0;
    /// Apply projective constraints to the given vector
    virtual void projectResponse(VecId dx, double **W=NULL) = 0;
    virtual void addMdx(VecId res, VecId dx, double factor) = 0; ///< res += M.dx
    virtual void integrateVelocity(VecId res, VecId x, VecId v, double dt) = 0; ///< res = x + v.dt
    virtual void accFromF(VecId a, VecId f) = 0; ///< a = M^-1 . f
    /// Propagate the given state (time, position and velocity) through all mappings
    virtual void propagatePositionAndVelocity(double t, VecId x, VecId v) = 0;

    /// Compute the current force (given the latest propagated position and velocity)
    virtual void computeForce(VecId result) = 0;
    /// Compute the current force delta (given the latest propagated displacement)
    virtual void computeDf(VecId df) = 0;
    /// Compute the current force delta (given the latest propagated velocity)
    virtual void computeDfV(VecId df) = 0;
    /// Compute the acceleration corresponding to the given state (time, position and velocity)
    virtual void computeAcc(double t, VecId a, VecId x, VecId v) = 0;

    virtual void computeContactForce(VecId result) = 0;
    virtual void computeContactDf(VecId df) = 0;
    virtual void computeContactAcc(double t, VecId a, VecId x, VecId v) = 0;

    /// @}

    /// @name Matrix operations
    /// @{

    // BaseMatrix & BaseVector Computations
    virtual void addMBK_ToMatrix(defaulttype::BaseMatrix *A, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0) = 0;
    virtual void addMBKdx_ToVector(defaulttype::BaseVector *V, VecId dx, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0) = 0;
    virtual void getMatrixDimension(unsigned int * const, unsigned int * const) = 0;
    virtual void multiVector2BaseVector(VecId src, defaulttype::BaseVector *dest=NULL, unsigned int offset=0) = 0;
    virtual void multiVectorPeqBaseVector(VecId dest, defaulttype::BaseVector *src=NULL, unsigned int offset=0) = 0;

    /// @}


    /// @name Matrix operations using LinearSolver components
    /// @{

    virtual void m_resetSystem() = 0;
    virtual void m_setSystemMBKMatrix(double mFact, double bFact, double kFact) = 0;
    virtual void m_setSystemRHVector(VecId v) = 0;
    virtual void m_setSystemLHVector(VecId v) = 0;
    virtual void m_solveSystem() = 0;
    virtual void m_print( std::ostream& out ) = 0;

    /// @}

    /// @name Debug operations
    /// @{

    /// Dump the content of the given vector.
    virtual void print( VecId v, std::ostream& out ) = 0;
    virtual void printWithElapsedTime( VecId v,  unsigned time, std::ostream& out=std::cerr ) = 0;

    /// @}

    /// @}

protected:
#if 0
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
#endif

    /// Helper class providing a high-level view of underlying state vectors.
    ///
    /// It is used to convert math-like operations to call to computation methods.
    class MultiVector
    {
    public:
        typedef OdeSolver::VecId VecId;

    protected:
        /// Solver who is using this vector
        OdeSolver* parent;

        /// Identifier of this vector
        VecId v;

        /// Flag indicating if this vector was dynamically allocated
        bool dynamic;

        /// Copy-constructor is forbidden
        MultiVector(const MultiVector& v);

    public:
        /// Refers to a state vector with the given ID (VecId::position(), VecId::velocity(), etc).
        MultiVector(OdeSolver* parent, VecId v) : parent(parent), v(v), dynamic(false)
        {}

        /// Allocate a new temporary vector with the given type (VecId::V_COORD or VecId::V_DERIV).
        MultiVector(OdeSolver* parent, VecId::Type t) : parent(parent), v(parent->v_alloc(t)), dynamic(true)
        {}

        ~MultiVector()
        {
            if (dynamic) parent->v_free(v);
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

        /// nullify values below given threshold
        void threshold( double threshold )
        {
            parent->v_threshold(v, threshold);
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

    /// Helper class allowing to construct mechanical expressions
    ///
    class MechanicalMatrix
    {
    protected:
        enum { MFACT = 0, BFACT = 1, KFACT = 2 };
        defaulttype::Vec<3,double> factors;
    public:
        MechanicalMatrix(double m, double b, double k) : factors(m,b,k) {}
        explicit MechanicalMatrix(const defaulttype::Vec<3,double>& f) : factors(f) {}

        double getMFact() const { return factors[MFACT]; }
        double getBFact() const { return factors[BFACT]; }
        double getKFact() const { return factors[KFACT]; }

        MechanicalMatrix operator + (const MechanicalMatrix& m2) { return MechanicalMatrix(factors + m2.factors); }
        MechanicalMatrix operator - (const MechanicalMatrix& m2) { return MechanicalMatrix(factors - m2.factors); }
        MechanicalMatrix operator - () { return MechanicalMatrix(- factors); }
        MechanicalMatrix operator * (double f) { return MechanicalMatrix(factors * f); }
        friend MechanicalMatrix operator * (double f, const MechanicalMatrix& m1) { return MechanicalMatrix(f * m1.factors); }
        MechanicalMatrix operator / (double f) { return MechanicalMatrix(factors / f); }
        friend std::ostream& operator << (std::ostream& out, const MechanicalMatrix& m )
        {
            out << '(';
            bool first = true;
            for (unsigned int i=0; i<m.factors.size(); ++i)
            {
                double f = m.factors[i];
                if (f!=0.0)
                {
                    if (!first) out << ' ';
                    if (f == -1.0) out << '-';
                    else if (f < 0) out << f << ' ';
                    else
                    {
                        if (!first) out << '+';
                        if (f != 1.0) out << f << ' ';
                    }
                    out << ("MBK")[i];
                    first = false;
                }
            }
            out << ')';
            return out;
        }
    };

    static const MechanicalMatrix M;
    static const MechanicalMatrix B;
    static const MechanicalMatrix K;

    /// Helper class providing a high-level view of underlying linear system matrices.
    ///
    /// It is used to convert math-like operations to call to computation methods.
    class MultiMatrix
    {
    public:
        typedef OdeSolver::VecId VecId;

    protected:
        /// Solver who is using this vector
        OdeSolver* parent;

        /// Identifier of this vector
        VecId v;

        /// Copy-constructor is forbidden
        MultiMatrix(const MultiVector& v);

    public:

        MultiMatrix(OdeSolver* parent) : parent(parent)
        {
        }

        ~MultiMatrix()
        {
        }

        /// m = 0
        void clear()
        {
            parent->m_resetSystem();
        }

        /// m = 0
        void reset()
        {
            parent->m_resetSystem();
        }

        /// m = m*M+b*B+k*K
        void operator=(const MechanicalMatrix& m)
        {
            parent->m_setSystemMBKMatrix(m.getMFact(), m.getBFact(), m.getKFact());
        }

        void solve(MultiVector& solution, MultiVector& rh)
        {
            parent->m_setSystemRHVector(rh);
            parent->m_setSystemLHVector(solution);
            parent->m_solveSystem();
        }
        friend std::ostream& operator << (std::ostream& out, const MultiMatrix& m )
        {
            m.parent->m_print(out);
            return out;
        }
    };
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
