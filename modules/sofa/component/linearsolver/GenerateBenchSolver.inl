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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_LINEARSOLVER_GENERATEBENCHSOLVER_INL
#define SOFA_COMPONENT_LINEARSOLVER_GENERATEBENCHSOLVER_INL

#include "GenerateBenchSolver.h"
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.inl>

//Random non zero
#define MAX_VALUE 20.0
#define DOMINANT_DIAGONAL 10000.0
#define RAND (((double) (((double) rand()+1.0)/((double) RAND_MAX+1.0)) * MAX_VALUE))

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;


template<class TMatrix, class TVector>
GenerateBenchSolver<TMatrix,TVector>::GenerateBenchSolver()
    : dump_system( initData(&dump_system,false,"dump_system","Dump the system at the next time step") )
    , file_system( initData(&file_system,std::string("file_system"),"file_system","Filename for the system") )
    , dump_constraint( initData(&dump_constraint,false,"dump_constraint","Dump the Jmatrix at the next time step") )
    , file_constraint( initData(&file_constraint,std::string("file_constraint"),"file_constraint","Filename for the J matrix") )
    , f_one_step( initData(&f_one_step,true,"one_step","Generate the matrix only for the next step else erase the file each step") )
{
}

template<class TMatrix, class TVector>
void GenerateBenchSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    if (dump_system.getValue())
    {
        if (f_one_step.getValue())
        {
            bool * dump = dump_system.beginEdit();
            dump[0] = false;
            dump_system.endEdit();
        }

        std::ofstream file(file_system.getValue().c_str(), std::fstream::out | std::fstream::binary);

        sofa::component::linearsolver::CompressedRowSparseMatrix<Real> Mfiltered;

        //M.compress();
        Mfiltered.resize(M.rowSize(),M.colSize());
        Mfiltered.copyNonZeros(M);
        Mfiltered.fullRows();
        Mfiltered.compress();

        int size = Mfiltered.rowSize();
        int size_type = sizeof(Real);
        int n_val = Mfiltered.getRowBegin()[size];

        if (this->f_printLog.getValue()) std::cout << "Mfiltered.getColsValue contains " << n_val << " non zero values" << std::endl;

        file.write((char*)&size, sizeof(int));
        file.write((char*)&size_type, sizeof(int));
        file.write((char*)&n_val, sizeof(int));
        file.write((char*)&(Mfiltered.getRowBegin()[0]), (size+1) * sizeof(int));
        file.write((char*)&(Mfiltered.getColsIndex()[0]), n_val * sizeof(int));
        file.write((char*)&(Mfiltered.getColsValue()[0]), n_val * sizeof(Real));

        file.write((char*)&(r[0]), size * sizeof(Real));
        file.write((char*)&(z[0]), size * sizeof(Real));

        file.close();

        std::cout << "File " << file_system.getValue() << " has been saved size of M is (" << M.rowSize() << "," << M.colSize() << ")" << std::endl;
    }
    z = r;
}

template<class TMatrix, class TVector> template<class RMatrix, class JMatrix>
bool GenerateBenchSolver<TMatrix,TVector>::addJMInvJt(RMatrix& /*result*/, JMatrix& J, double fact)
{
    if (dump_constraint.getValue())
    {
        if (f_one_step.getValue())
        {
            bool * dump = dump_constraint.beginEdit();
            dump[0] = false;
            dump_constraint.endEdit();
        }

        std::ofstream file(file_constraint.getValue().c_str(), std::fstream::out | std::fstream::binary);
        //file.seekp(0, std::ios::end);


        int szLin = 0;
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            if (jit1->second.size()>0) szLin++;
        }

        int sizeX = J.colSize();
        int sizeY = J.rowSize();
        int size_type = sizeof(Real);
        int * cols = (int*) malloc(sizeX * sizeof(int));
        Real * vals = (Real*) malloc(sizeX * sizeof(Real));

        file.write((char*)&sizeX, sizeof(int));
        file.write((char*)&sizeY, sizeof(int));
        file.write((char*)&szLin, sizeof(int));
        file.write((char*)&size_type, sizeof(int));
        file.write((char*)&fact, sizeof(double));

        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            int lin = jit1->first;
            int szlin = jit1->second.size();
            int n_val=0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
            {
                cols[n_val] = i1->first;
                vals[n_val] = i1->second;
                n_val++;
            }

            file.write((char*)&lin, sizeof(int));
            file.write((char*)&szlin, sizeof(int));
            file.write((char*)cols, szlin * sizeof(int));
            file.write((char*)vals, szlin * sizeof(Real));
        }

        file.close();
        free(cols);
        free(vals);

        if (this->f_printLog.getValue()) std::cout << J << std::endl;

        std::cout << "File " << file_constraint.getValue() << " has been saved size of J is (" << J.rowSize() << "," << J.colSize() << ")" << std::endl;
    }
    return true;
}

template<class TMatrix, class TVector>
bool GenerateBenchSolver<TMatrix,TVector>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
    {
        return addJMInvJt(*result,*j,fact);
    }
    else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
    {
        return addJMInvJt(*result,*j,fact);
    }
    else
    {
        serr << "ERROR : Unknown matrix format in ParallelMatrixLinearSolver<Matrix,Vector>::addJMInvJt" << sendl;
        return false;
    }
}

template<class TMatrix, class TVector>
bool GenerateBenchSolver<TMatrix,TVector>::read_system(int & max_size,std::string & fileName,TMatrix & matrix,sofa::defaulttype::BaseVector * solution,sofa::defaulttype::BaseVector * unknown,bool print)
{
    std::ifstream file(fileName.c_str(), std::ifstream::binary);

    if (file.good())
    {
        int size_type;
        int n_val;
        int size;
        file.read((char*) &size, sizeof(int));
        if (max_size==0) max_size=size;
        else if (max_size>size)
        {
            std::cerr << "Warning the file has a maximum size of " << size << std::endl;
            max_size=size;
        }

        file.read((char*) &size_type, sizeof(int));
        file.read((char*) &n_val, sizeof(int));

        if (size_type!=sizeof(Real))
        {
            std::cerr << "Error the bench has been created with another type Real" << std::endl;
            file.close();
            return false;
        }

        if (print) std::cout << "file open : " << fileName << " size = " << size << " syze_type=" << size_type << " nb_val=" << n_val << std::endl;

        int * row_ind = (int*)  malloc((size+1)* sizeof(int));
        int * col_ind = (int*)  malloc( n_val  * sizeof(int));
        Real * values = (Real*) malloc( n_val  * sizeof(Real));
        Real * sol = (Real*)  malloc(size* sizeof(Real));
        Real * unk = (Real*)  malloc(size* sizeof(Real));

        file.read((char*)row_ind,(size+1)* sizeof(int));
        file.read((char*)col_ind, n_val  * sizeof(int));
        file.read((char*)values , n_val  * sizeof(Real));

        file.read((char*)sol, size * sizeof(Real));
        file.read((char*)unk, size * sizeof(Real));

        file.close();

        if (n_val!=row_ind[size])
        {
            std::cerr << "Error there was a problem in the generation of the matrix col_ind[size]=" << col_ind[size] << " nval=" << n_val << std::endl;
            return false;
        }

        matrix.resize(max_size,max_size);
        solution->resize(max_size);
        unknown->resize(max_size);
        for (int j=0; j<max_size; j++)
        {
            solution->set(j,sol[j]);
            unknown->set(j,unk[j]);
            for (int i=row_ind[j]; i<row_ind[j+1]; i++)
            {
                if (col_ind[i]<max_size) matrix.set(j,col_ind[i],values[i]);
            }
        }

        matrix.compress();

        //if (print) std::cout << matrix << std::endl;

        free(row_ind);
        free(col_ind);
        free(values);
        free(sol);
        free(unk);

        if (print) std::cout << "System is loaded" << std::endl;

        return  true;
    }

    std::cerr << "The file " << fileName.c_str() << " doesn't exist" << std::endl;

    return false;
}

template<class TMatrix, class TVector> template<class JMatrix>
bool GenerateBenchSolver<TMatrix,TVector>::read_J(int & max_size,int size,std::string & fileName,JMatrix & J,double & fact,bool print)
{
    std::ifstream file(fileName.c_str(), std::ifstream::binary);

    if (file.good())
    {
        int size_type;
        int sizeX;
        int sizeY;
        int szLin;
        file.read((char*) &sizeX, sizeof(int));
        file.read((char*) &sizeY, sizeof(int));
        file.read((char*) &szLin, sizeof(int));
        file.read((char*) &size_type, sizeof(int));
        file.read((char*) &fact, sizeof(double));

        if (sizeX!= (int) size)
        {
            std::cerr << "Error the J matrix has not a valid dimention J.size=" << sizeX << " System.size=" << size << std::endl;
            return false;
        }

        if (sizeY < max_size) std::cerr << "WARNING the J matrix contains only " << sizeY << std::endl;
        max_size = sizeY;

        if (size_type!=sizeof(Real))
        {
            std::cerr << "Error the bench has been created with another type Real" << std::endl;
            return false;
        }

        if (print) std::cout << "file open : " << fileName << " sizeX = " << sizeX << " syzeY=" << sizeY << std::endl;

        J.resize(sizeY,sizeX);
        int * cols = (int*) malloc(size * sizeof(int));
        Real * vals = (Real*) malloc(size * sizeof(Real));

        for (int l=0; l<szLin; l++)
        {
            int lin;
            int nval;
            file.read((char*)&lin,sizeof(int));
            file.read((char*)&nval,sizeof(int));
            file.read((char*)cols , nval  * sizeof(int));
            file.read((char*)vals , nval  * sizeof(Real));

            for (int i=0; i<nval; i++) J.set(lin,cols[i],vals[i]);
        }

        //if (print) std::cout << J << std::endl;

        if (print) std::cout << "J is loaded" << std::endl;

        return true;
    }

    std::cerr << "The file " << fileName.c_str() << " doesn't exist" << std::endl;

    return false;
}


template<class TMatrix, class TVector>
bool GenerateBenchSolver<TMatrix,TVector>::generate_system(int & size,double sparsity,TMatrix & matrix,sofa::defaulttype::BaseVector * solution,sofa::defaulttype::BaseVector * unknown,bool print)
{
    if (size==0)
    {
        std::cerr << "Warning you should specify the size default value is use : 1002 " << std::endl;
        size=1002;
    }

    matrix.resize(size,size);
    solution->resize(size);
    unknown->resize(size);

    srand(MAX_VALUE);
    int taux = 0;
    double t;
    for (int y=0; y<size; y+=3)
    {
        matrix.set(y,y,RAND*DOMINANT_DIAGONAL); matrix.set(y+1,y+1,RAND*DOMINANT_DIAGONAL); matrix.set(y+2,y+2,RAND*DOMINANT_DIAGONAL);
        t=RAND; matrix.set(y,y+1,t); matrix.set(y+1,y,t);
        t=RAND; matrix.set(y,y+2,t); matrix.set(y+2,y,t);
        t=RAND; matrix.set(y+1,y+2,t); matrix.set(y+2,y+1,t);
        taux+=9;

        for (int x=y+3; x<size; x+=3)
        {
            double t = RAND;
            if (t<MAX_VALUE * sparsity / 100.0)
            {
                matrix.set(y,  x  ,t); matrix.set(x  ,y  ,t);
                t=RAND; matrix.set(y,  x+1,t); matrix.set(x+1,y  ,t);
                t=RAND; matrix.set(y,  x+2,t); matrix.set(x+2,y  ,t);
                t=RAND; matrix.set(y+1,x  ,t); matrix.set(x,  y+1,t);
                t=RAND; matrix.set(y+1,x+1,t); matrix.set(x+1,y+1,t);
                t=RAND; matrix.set(y+1,x+2,t); matrix.set(x+2,y+1,t);
                t=RAND; matrix.set(y+2,x  ,t); matrix.set(x  ,y+2,t);
                t=RAND; matrix.set(y+2,x+1,t); matrix.set(x+1,y+2,t);
                t=RAND; matrix.set(y+2,x+2,t); matrix.set(x+2,y+2,t);

                taux+=18;
            }
        }

        unknown->set(y+0,RAND);
        unknown->set(y+1,RAND);
        unknown->set(y+2,RAND);
    }

    for (int y=0; y<size; y++)
    {
        double acc = 0.0;
        for (int x=0; x<size; x++) acc += matrix.element(y,x) * unknown->element(x);
        solution->set(y,acc);
    }

//   std::cout << matrix << std::endl;

//   std::cout << matrix << std::endl;
//   std::cout << solution << std::endl;
    if (print) std::cout << "Sparsity of the generated matrix is " << ((taux*100.0)/((double) (size*size))) << "%" << std::endl;

    return true;
}


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
