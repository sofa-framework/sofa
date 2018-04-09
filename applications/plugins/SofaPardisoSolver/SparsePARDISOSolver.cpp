/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaPardisoSolver/SparsePARDISOSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.inl>

#include <sys/types.h>
#include <sys/stat.h>

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif
#include <stdlib.h>

/* Change this if your Fortran compiler does not append underscores. */
/* e.g. the AIX compiler:  #define F77_FUNC(func) func                */

#ifdef AIX
#define F77_FUNC(func)  func
#else
#define F77_FUNC(func)  func ## _
#endif
extern "C" {

    /* PARDISO prototype. */
    extern  int F77_FUNC(pardisoinit)
    (void *, int *, int *, int *, double *, int *);

    extern  int F77_FUNC(pardiso)
    (void *, int *, int *, int *, int *, int *,
     double *, int *, int *, int *, int *, int *,
     int *, double *, double *, int *, double *);

} // "C"


namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TMatrix, class TVector>
SparsePARDISOSolver<TMatrix,TVector>::SparsePARDISOSolverInvertData::SparsePARDISOSolverInvertData(int f_symmetric,std::ostream & sout,std::ostream & serr)
    : solver(NULL)
    , pardiso_initerr(1)
    , pardiso_mtype(0)
    , factorized(false)
{
    factorized = false;
    pardiso_initerr = 0;

    std::cout << "FSYM: " << f_symmetric << std::endl;

    switch(f_symmetric)
    {
    case  0: pardiso_mtype = 11; break; // real and nonsymmetric
    case  1: pardiso_mtype = -2; break; // real and symmetric indefinite
    case  2: pardiso_mtype =  2; break; // real and symmetric positive definite
    case -1: pardiso_mtype =  1; break; // real and structurally symmetric
    default:
        pardiso_mtype = 11; break; // real and nonsymmetric
    }
    pardiso_iparm[0] = 0;
    int solver = 0; /* use sparse direct solver */
    /* Numbers of processors, value of OMP_NUM_THREADS */
    const char* var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        pardiso_iparm[2] = atoi(var);
    else
        pardiso_iparm[2] = 1;
    sout << "Using " << pardiso_iparm[2] << " thread(s), set OMP_NUM_THREADS environment variable to change." << std::endl;

    F77_FUNC(pardisoinit) (pardiso_pt,  &pardiso_mtype, &solver, pardiso_iparm, pardiso_dparm, &pardiso_initerr);

    switch(pardiso_initerr)
    {
    case 0:   sout << "PARDISO: License check was successful" << std::endl; break;
    case -10: serr << "PARDISO: No license file found" << std::endl; break;
    case -11: serr << "PARDISO: License is expired" << std::endl; break;
    case -12: serr << "PARDISO: Wrong username or hostname" << std::endl; break;
    default:  serr << "PARDISO: Unknown error " << pardiso_initerr << std::endl; break;
    }
    //if (data->pardiso_initerr) return;
    //if(var != NULL)
    //    data->pardiso_iparm[2] = atoi(var);
    //else
    //    data->pardiso_iparm[2] = 1;

}


template<class TMatrix, class TVector>
SparsePARDISOSolver<TMatrix,TVector>::SparsePARDISOSolver()
    : f_symmetric( initData(&f_symmetric,1,"symmetric","0 = nonsymmetric arbitrary matrix, 1 = symmetric matrix, 2 = symmetric positive definite, -1 = structurally symmetric matrix") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_exportDataToDir( initData(&f_exportDataToDir, std::string(""), "exportDataToDir", "export data (matrix, RHS, solution) to files in given directory"))
    , f_iterativeSolverNumbering( initData(&f_iterativeSolverNumbering,false,"iterativeSolverNumbering","if true, the naming convention is incN_itM where N is the time step and M is the iteration inside the step") )
    , f_saveDataToFile( initData(&f_saveDataToFile,false,"saveDataToFile","if true, export the data to the current directory (if exportDataToDir not set)") )
{
}

template<class TMatrix, class TVector>
void SparsePARDISOSolver<TMatrix,TVector>::init()
{
    numStep = 0;
    numPrevNZ = 0;
    numActNZ = 0;
    Inherit::init();

    std::string exportDir=f_exportDataToDir.getValue();

    doExportData = false;
    if (!exportDir.empty()) {
        int status;
        status = mkdir(exportDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0 || errno == EEXIST)
            doExportData = true;
        else
            std::cout << this->getName() << " WARNING: cannot create directory " << exportDir << " for exporting the solver data" << std::endl;

    } else {
        if (f_saveDataToFile.getValue()) {
            f_exportDataToDir.setValue(".");
            doExportData = 1;
        }
    }
    timeStep = -1;
}

template<class TMatrix, class TVector>
SparsePARDISOSolver<TMatrix,TVector>::~SparsePARDISOSolver()
{
}

template<class TMatrix, class TVector>
int SparsePARDISOSolver<TMatrix,TVector>::callPardiso(SparsePARDISOSolverInvertData* data, int phase, Vector* vx, Vector* vb)
{
    int maxfct = 1; // Maximum number of numerical factorizations
    int mnum = 1; // Which factorization to use
    int n = data->Mfiltered.rowSize();
    double* a = NULL;
    int* ia = NULL;
    int* ja = NULL;
    int* perm = NULL; // User-specified permutation vector
    int nrhs = 0; // Number of right hand sides
    int msglvl = (f_verbose.getValue())?1:0; // Print statistical information
    double* b = NULL;
    double* x = NULL;
    int error = 0;

    if (phase > 0)
    {
        ia = (int *) &(data->Mfiltered.getRowBegin()[0]);
        ja = (int *) &(data->Mfiltered.getColsIndex()[0]);
        a  = (double*) &(data->Mfiltered.getColsValue()[0]);

        //numActNZ = ia[n]-1;

        if (doExportData) {
            std::string exportDir=f_exportDataToDir.getValue();
            std::ofstream f;            
            char name[100];
            sprintf(name, "%s/spmatrix_PARD_%s.txt", exportDir.c_str(), suffix.c_str());
            f.open(name);

            int rw = 0;
            for (int i = 0; i < ia[n]-1; i++) {
                if (ia[rw] == i+1)
                    rw++;
                f << rw << " " << ja[i] << " " << a[i] << std::endl;
            }

            f.close();

            sprintf(name, "%s/compmatrix_PARD_%s.txt", exportDir.c_str(), suffix.c_str());
            f.open(name);

            for (int i = 0; i <= n; i++)
                f << ia[i] << std::endl;

            for (int i = 0; i < ia[n]-1; i++)
                f << ja[i] << " " << a[i] << std::endl;

            f.close();

        }

        if (vx)
        {
            nrhs = 1;
            x = vx->ptr();
            b = vb->ptr();
        }
    }
    sout << "Solver phase " << phase << "..." << sendl;
    sofa::helper::AdvancedTimer::stepBegin("PardisoRealSolving");
    F77_FUNC(pardiso)(data->pardiso_pt, &maxfct, &mnum, &data->pardiso_mtype, &phase,
            &n, a, ia, ja, perm, &nrhs,
            data->pardiso_iparm, &msglvl, b, x, &error,  data->pardiso_dparm);
    sofa::helper::AdvancedTimer::stepEnd("PardisoRealSolving");
    const char* msg = NULL;
    switch(error)
    {
    case 0: break;
    case -1: msg="Input inconsistent"; break;
    case -2: msg="Not enough memory"; break;
    case -3: msg="Reordering problem"; break;
    case -4: msg="Zero pivot, numerical fact. or iterative refinement problem"; break;
    case -5: msg="Unclassified (internal) error"; break;
    case -6: msg="Preordering failed (matrix types 11, 13 only)"; break;
    case -7: msg="Diagonal matrix problem"; break;
    case -8: msg="32-bit integer overflow problem"; break;
    case -10: msg="No license file pardiso.lic found"; break;
    case -11: msg="License is expired"; break;
    case -12: msg="Wrong username or hostname"; break;
    case -100: msg="Reached maximum number of Krylov-subspace iteration in iterative solver"; break;
    case -101: msg="No sufficient convergence in Krylov-subspace iteration within 25 iterations"; break;
    case -102: msg="Error in Krylov-subspace iteration"; break;
    case -103: msg="Break-Down in Krylov-subspace iteration"; break;
    default: msg="Unknown error"; break;
    }
    if (msg)
        serr << "Solver phase " << phase << ": ERROR " << error << " : " << msg << sendl;
    return error;
}

template<class TMatrix, class TVector>
void SparsePARDISOSolver<TMatrix,TVector>::invert(Matrix& M)
{
    if (f_iterativeSolverNumbering.getValue()) {
        double dt = Inherit::getContext()->getDt();
        double actualTime = Inherit::getContext()->getTime();

        long int actTimeStep = lround(actualTime/dt);

        if (timeStep != actTimeStep)
            numStep = 0;
        timeStep = actTimeStep;
        char nm[100];
        sprintf(nm, "step%04ld_iter%04d", timeStep, numStep);
        suffix = nm;
    } else {
        char nm[100];
        sprintf(nm, "%04d", numStep);
        suffix = nm;
    }



    sofa::helper::AdvancedTimer::stepBegin("PardisoInvert");

    if (doExportData) {
        std::string exportDir=f_exportDataToDir.getValue();
        std::ofstream f;
        char name[100];
        sprintf(name, "%s/matrix_PARD_%s.txt", exportDir.c_str(), suffix.c_str());
        //std::cout << this->getName() << ": Exporting to " << name << std::endl;
        f.open(name);
        f << M;
        f.close();
    }

    M.compress();    

    SparsePARDISOSolverInvertData * data = (SparsePARDISOSolverInvertData *) this->getMatrixInvertData(&M);

    if (data->pardiso_initerr) return;
    data->Mfiltered.clear();
    if (f_symmetric.getValue() > 0)
    {
        data->Mfiltered.copyUpperNonZeros(M);
        data->Mfiltered.fullDiagonal();
        sout << "Filtered upper part of M, nnz = " << data->Mfiltered.getRowBegin().back() << sendl;
    }
    else if (f_symmetric.getValue() < 0)
    {
        data->Mfiltered.copyNonZeros(M);
        data->Mfiltered.fullDiagonal();
        sout << "Filtered M, nnz = " << data->Mfiltered.getRowBegin().back() << sendl;
    }
    else
    {
        data->Mfiltered.copyNonZeros(M);
        data->Mfiltered.fullRows();
        sout << "Filtered M, nnz = " << data->Mfiltered.getRowBegin().back() << sendl;
    }
    //  Convert matrix from 0-based C-notation to Fortran 1-based notation.
    data->Mfiltered.shiftIndices(1);

    /* -------------------------------------------------------------------- */
    /* ..  Reordering and Symbolic Factorization.  This step also allocates */
    /*     all memory that is necessary for the factorization.              */
    /* -------------------------------------------------------------------- */

    numActNZ = data->Mfiltered.getRowBegin().back();
    //std::cout << this->getName() << "Actual NNZ = " << numActNZ << " previous NNZ = " << numPrevNZ << std::endl;
    if (numPrevNZ != numActNZ)
    {
        //std::cout << "[" << this->getName() << "] analyzing the matrix" << std::endl;
        //sout << "Analyzing the matrix" << std::endl;
        if (callPardiso(data, 11)) return;
        data->factorized = true;        
        //sout << "Reordering completed ..." << sendl;
        std::cout << "After analysis: NNZ = " << data->pardiso_iparm[17] << std::endl;
        //sout << "Number of factorization MFLOPS = " << data->pardiso_iparm[18] << sendl;
    }
    numPrevNZ = numActNZ;

    /* -------------------------------------------------------------------- */
    /* ..  Numerical factorization.                                         */
    /* -------------------------------------------------------------------- */
    std::cout << "[" << this->getName() << "] factorize the matrix" << std::endl;
    if (callPardiso(data, 22)) { data->factorized = false; return; }    

    sout << "Factorization completed ..." << sendl;
    sofa::helper::AdvancedTimer::stepEnd("PardisoInvert");

    numStep++;
}

template<class TMatrix, class TVector>
void SparsePARDISOSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{        
    if (doExportData){
        std::string exportDir=f_exportDataToDir.getValue();
        std::ofstream f;
        char name[100];
        sprintf(name, "%s/rhs_PARD_%s.txt", exportDir.c_str(), suffix.c_str());
        f.open(name);
        f << r;
        f.close();
    }

    sofa::helper::AdvancedTimer::stepBegin("PardisoSolve");
    SparsePARDISOSolverInvertData * data = (SparsePARDISOSolverInvertData *) this->getMatrixInvertData(&M);

    if (data->pardiso_initerr) return;
    if (!data->factorized) return;

    /* -------------------------------------------------------------------- */
    /* ..  Back substitution and iterative refinement.                      */
    /* -------------------------------------------------------------------- */
    data->pardiso_iparm[7] = 0;       /* Max numbers of iterative refinement steps. */

    //std::cout << "[" << this->getName() << "] solve the matrix" << std::endl;
    if (callPardiso(data, 33, &z, &r)) return;
    sofa::helper::AdvancedTimer::stepEnd("PardisoSolve");

    if (doExportData){
        std::string exportDir=f_exportDataToDir.getValue();
        std::ofstream f;
        char name[100];
        sprintf(name, "%s/solution_PARD_%s.txt", exportDir.c_str(), suffix.c_str());
        f.open(name);
        f << z;
        f.close();
    }    
}


SOFA_DECL_CLASS(SparsePARDISOSolver)

int SparsePARDISOSolverClass = core::RegisterObject("Direct linear solvers implemented with the PARDISO library")
        .add< SparsePARDISOSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >()
        .add< SparsePARDISOSolver< CompressedRowSparseMatrix< defaulttype::Mat<3,3,double> >,FullVector<double> > >(true)
        .addAlias("PARDISOSolver")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

