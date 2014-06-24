/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <SofaBaseLinearSolver/BaseLinearSolver.h>
#include <sofa/core/Plugin.h>

#include <SofaBaseLinearSolver/BTDLinearSolver.inl>
#include <SofaBaseLinearSolver/CGLinearSolver.inl>
#include <SofaBaseLinearSolver/CholeskySolver.inl>
#include <SofaBaseLinearSolver/MinResLinearSolver.inl>

using namespace sofa::component::linearsolver;
using namespace sofa::defaulttype;

class BaseLinearSolverPlugin: public sofa::core::Plugin {
public:
    BaseLinearSolverPlugin(): Plugin("BaseLinearSolver") {
        setDescription("");
        setVersion("");
        setLicense("LGPL");
        setAuthors("The SOFA Team");

        // Default template instance for BTDLinearSolver
#ifdef SOFA_FLOAT
        addComponent< BTDLinearSolver<BTDMatrix<6,float>,BlockVector<6,float> > >();
#else
        addComponent< BTDLinearSolver<BTDMatrix<6,double>,BlockVector<6,double> > >();
#endif
        setDescription("BTDLinearSolver", "Linear system solver using Thomas Algorithm for Block Tridiagonal matrices");
        // Other template instances for BTDLinearSolver
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< BTDLinearSolver<BTDMatrix<6,float>,BlockVector<6,float> > >();
#endif

        addComponent< CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector > >("Linear system solver using the conjugate gradient iterative algorithm");
        addTemplateInstance< CGLinearSolver< FullMatrix<double>, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< SparseMatrix<double>, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<2,2,double> >, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<2,2,float> >, FullVector<float> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<3,3,double> >, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<3,3,float> >, FullVector<float> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<4,4,double> >, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<4,4,float> >, FullVector<float> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<6,6,double> >, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<6,6,float> >, FullVector<float> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<8,8,double> >, FullVector<double> > >();
        addTemplateInstance< CGLinearSolver< CompressedRowSparseMatrix<Mat<8,8,float> >, FullVector<float> > >();
        addAlias("CGLinearSolver", "CGSolver");
        addAlias("CGLinearSolver", "ConjugateGradient");

        addComponent< CholeskySolver< SparseMatrix<double>, FullVector<double> > >("Direct linear solver based on Cholesky factorization, for dense matrices");
        addTemplateInstance< CholeskySolver< FullMatrix<double>, FullVector<double> > >();
        addTemplateInstance< CholeskySolver< FullMatrix<float>, FullVector<float> > >();


        addComponent< MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector > >("Linear system solver using the MINRES iterative algorithm");
        addTemplateInstance< MinResLinearSolver< FullMatrix<double>, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< SparseMatrix<double>, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,double> >, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,float> >, FullVector<float> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,double> >, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,float> >, FullVector<float> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,double> >, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,float> >, FullVector<float> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,double> >, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,float> >, FullVector<float> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,double> >, FullVector<double> > >();
        addTemplateInstance< MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,float> >, FullVector<float> > >();
        addAlias("MinResLinearSolver", "MINRESSolver");
        addAlias("MinResLinearSolver", "MinResSolver");

    }
};

SOFA_PLUGIN(BaseLinearSolverPlugin);
