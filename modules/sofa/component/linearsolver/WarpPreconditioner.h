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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_H
#define SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_H

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/linearsolver/FullVector.h>
#include <math.h>

#include <map>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TDataTypes>
class WarpPreconditionerInternalData
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef FullVector<Real> TBaseVector ;


};

/// Linear system solver wrapping another (precomputed) linear solver by a per-node rotation matrix
template<class TDataTypes>
class WarpPreconditioner : public core::behavior::LinearSolver
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(WarpPreconditioner,TDataTypes),sofa::core::behavior::LinearSolver);
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename WarpPreconditionerInternalData<DataTypes>::TBaseVector TBaseVector;
    typedef sofa::defaulttype::MatNoInit<3, 3, Real> Transformation;
    typedef sofa::core::behavior::LinearSolver Inherit;

    Data<bool> f_verbose;
    Data <std::string> solverName;

    WarpPreconditioner();
    //void solve (TMatrix& M, TVector& x, TVector& b);
    //void invert(TMatrix& M);
    //void setSystemMBKMatrix(double mFact=0.0, double bFact=0.0, double kFact=0.0);
    //bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact);
    void bwdInit();
    //void loadMatrix();

    /// Reset the current linear system.
    virtual void resetSystem();

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    virtual void setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams);

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    virtual void setSystemRHVector(core::MultiVecDerivId v);

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    virtual void setSystemLHVector(core::MultiVecDerivId v);

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem();

    /// Invert the system, this method is optional because it's call when solveSystem() is called for the first time
    virtual void invertSystem();

    /// Multiply the inverse of the system matrix by the transpose of the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
    { if (realSolver) return realSolver->addMInvJt(result, J, fact); else return false; }

    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
    { if (realSolver) return realSolver->addJMInvJt(result, J, fact); else return false; }

    /// Get the linear system matrix, or NULL if this solver does not build it
    virtual defaulttype::BaseMatrix* getSystemBaseMatrix()
    { if (realSolver) return realSolver->getSystemBaseMatrix(); else return NULL; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    virtual defaulttype::BaseVector* getSystemRHBaseVector()
    { if (realSolver) return realSolver->getSystemRHBaseVector(); else return NULL; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    virtual defaulttype::BaseVector* getSystemLHBaseVector()
    { if (realSolver) return realSolver->getSystemLHBaseVector(); else return NULL; }

    /// Get the linear system inverse matrix, or NULL if this solver does not build it
    virtual defaulttype::BaseMatrix* getSystemInverseBaseMatrix()
    { if (realSolver) return realSolver->getSystemInverseBaseMatrix(); else return NULL; }

    /// Read the Matrix solver from a file
    virtual bool readFile(std::istream& in)
    { if (realSolver) return realSolver->readFile(in); else return false; }

    /// Read the Matrix solver from a file
    virtual bool writeFile(std::ostream& out)
    { if (realSolver) return realSolver->writeFile(out); else return false; }

    /// Ask the solver to no longer update the system matrix
    virtual void freezeSystemMatrix()
    {
        Inherit::freezeSystemMatrix();
        if (realSolver) realSolver->freezeSystemMatrix();
    }

    /// Ask the solver to no update the system matrix at the next iteration
    virtual void updateSystemMatrix()
    {
        Inherit::updateSystemMatrix();
        if (realSolver) realSolver->updateSystemMatrix();
    }


    //template<class JMatrix>
    //void ComputeCudaResult(CudaBaseMatrix<float>& result,JMatrix& J, float fact, bool localW);


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const WarpPreconditioner<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

private :
    //CudaBaseVector<cuda_real> CudaR;
    //CudaBaseVector<cuda_real> CudaT;

    core::behavior::LinearSolver* realSolver;
    core::behavior::MechanicalState<DataTypes>* mstate;
    component::forcefield::TetrahedronFEMForceField<DataTypes>* forceField;

    core::MultiVecDerivId systemLHVId;
    core::MultiVecDerivId systemRHVId;
    core::VecDerivId rotatedLHVId;
    core::VecDerivId rotatedRHVId;

    WarpPreconditionerInternalData<DataTypes> data;

    //CudaMatrixUtils Utils;

    void getRotations(TBaseVector & R);

    TBaseVector Rcurr;
    TBaseVector Rinv;
};


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
