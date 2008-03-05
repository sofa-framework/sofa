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
#ifndef SOFA_COMPONENT_LINEARSOLVER_LULINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_LULINEARSOLVER_H

#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/simulation/tree/MatrixLinearSolver.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Linear system solver using the default (LU factorization) algorithm
template<class Matrix, class Vector>
class LULinearSolver : public sofa::simulation::tree::MatrixLinearSolver<Matrix,Vector>, public virtual sofa::core::objectmodel::BaseObject
{
public:
    Data<bool> f_verbose;

    LULinearSolver()
        : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    {
    }

    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b)
    {
        using std::cerr;
        using std::endl;

        const bool verbose  = f_verbose.getValue();

        if( verbose )
        {
            cerr<<"LULinearSolver, b = "<< b <<endl;
            cerr<<"LULinearSolver, M = "<< M <<endl;
        }
        M.solve(&x,&b);
        // x is the solution of the system
        if( verbose )
        {
            cerr<<"LULinearSolver::solve, solution = "<<x<<endl;
        }
    }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
