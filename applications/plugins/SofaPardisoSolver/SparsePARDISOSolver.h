/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparsePARDISOSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_SparsePARDISOSolver_H
#include "config.h"

#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>

#include <assert.h>
#include <float.h>
#include <stdlib.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Direct linear solvers implemented with the PARDISO library
template<class TMatrix, class TVector>
class SparsePARDISOSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SparsePARDISOSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    //Data< helper::vector<std::string> > f_options;
    Data<int> f_symmetric;
    Data<bool> f_verbose;
    Data<std::string> f_exportDataToDir;
    Data<bool> f_iterativeSolverNumbering;
    Data<bool> f_saveDataToFile;

    SparsePARDISOSolver();
    ~SparsePARDISOSolver();
    virtual void init();

    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

    MatrixInvertData * createInvertData()
    {
        return new SparsePARDISOSolverInvertData(f_symmetric.getValue(),std::cout,std::cerr);
    }

protected:
    bool doExportData;
    std::string suffix;

    int numStep;
    long int timeStep;
    int numPrevNZ, numActNZ;
    class SparsePARDISOSolverInvertData : public MatrixInvertData
    {
    public :
        CompressedRowSparseMatrix<double> Mfiltered;
        SparsePARDISOSolver<Matrix,Vector>* solver;
        void*  pardiso_pt[64];
        int    pardiso_iparm[64];
        double pardiso_dparm[64];
        int pardiso_initerr;
        int pardiso_mtype;
        bool factorized;

        SparsePARDISOSolverInvertData(int f_symmetric,std::ostream & sout,std::ostream & serr);

        ~SparsePARDISOSolverInvertData()
        {
            if (solver && pardiso_initerr == 0)
            {
                solver->callPardiso(this, -1);  // Release internal memory.
            }
        }

    };


    int callPardiso(SparsePARDISOSolverInvertData* data, int phase, Vector* vx = NULL, Vector* vb = NULL);
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
