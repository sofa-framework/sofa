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
#ifndef SOFA_COMPONENT_LINEARSOLVER_GENERATEBENCHSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_GENERATEBENCHSOLVER_H

#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{


/// Direct linear solvers implemented with the TAUCS library
template<class TMatrix, class TVector>
class GenerateBenchSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(GenerateBenchSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    typedef core::VecId VecId;

    Data<bool> dump_system;
    Data< std::string >  file_system;
    Data<bool> dump_constraint;
    Data< std::string >  file_constraint;

    GenerateBenchSolver();
    void solve (Matrix& M, Vector& x, Vector& b);

    template<class RMatrix, class JMatrix>
    bool addJMInvJt(RMatrix& result, JMatrix& J, double fact);

    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact);

    static bool read_system(int & max_size,std::string & fileName,TMatrix & matrix,sofa::defaulttype::BaseVector * solution,sofa::defaulttype::BaseVector * unknown,bool print);
    template<class JMatrix>
    static bool read_J(int & max_size,int size,std::string & fileName,JMatrix & J,double & fact,bool print);

    static bool generate_system(int & max_size,double sparsity,TMatrix & matrix,sofa::defaulttype::BaseVector * solution,sofa::defaulttype::BaseVector * unknown,bool print);
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
