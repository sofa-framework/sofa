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
#include <sofa/component/linearsolver/iterative/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/map.h>

#include <cmath>

namespace sofa::component::linearsolver::iterative
{

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class ShewchukPCGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ShewchukPCGLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<unsigned> f_maxIter; ///< maximum number of iterations of the Conjugate Gradient solution
    Data<double> f_tolerance; ///< desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)
    Data<bool> f_use_precond; ///< Use preconditioner
    SingleLink<ShewchukPCGLinearSolver, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_preconditioner; ///< Link towards the linear solver used to precondition the conjugate gradient
    Data<unsigned> f_update_step; ///< Number of steps before the next refresh of precondtioners
    Data<bool> f_build_precond; ///< Build the preconditioners, if false build the preconditioner only at the initial step
    Data<std::map < std::string, sofa::type::vector<double> > > f_graph; ///< Graph of residuals at each iteration


    SOFA_ATTRIBUTE_DISABLED__SHEWCHUKPCGLINEARSOLVER_EXPLICITLINK()
    Data< std::string > f_preconditioners; ///< If not empty: path to the solvers to use as preconditioners
    //SOFA_ATTRIBUTE_DISABLED__SHEWCHUKPCGLINEARSOLVER_EXPLICITLINK()
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        Inherit1::parse(arg);
        if (arg->getAttribute("preconditioners"))
        {
            msg_warning() << "String data \"preconditioners\" is now replaced by explicit data link: \"preconditioner\" (PR #3155)";
        }
    }

protected:
    ShewchukPCGLinearSolver();

public:
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void init() override;
    void setSystemMBKMatrix(const core::MechanicalParams* mparams) override;

private :
    unsigned next_refresh_step;
    bool first;
    int newton_iter;

protected:
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: p = p*beta + r
    inline void cgstep_beta(Vector& p, Vector& r, double beta);
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(Vector& x,Vector& p,double alpha);

    void handleEvent(sofa::core::objectmodel::Event* event) override;


};

template<class TMatrix, class TVector>
inline void ShewchukPCGLinearSolver<TMatrix,TVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    p *= beta;
    p += r; //z;
}

template<class TMatrix, class TVector>
inline void ShewchukPCGLinearSolver<TMatrix,TVector>::cgstep_alpha(Vector& x,Vector& p,double alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, double beta);

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x,Vector& p,double alpha);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_SHEWCHUKPCGLINEARSOLVER_CPP)
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API ShewchukPCGLinearSolver<GraphScatteredMatrix, GraphScatteredVector>;
#endif // !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_SHEWCHUKPCGLINEARSOLVER_CPP)

} // namespace sofa::component::linearsolver::iterative
