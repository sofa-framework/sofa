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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenSparseSquareMatrix_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenSparseSquareMatrix_H

#include <sofa/defaulttype/BaseMatrix.h>
#ifdef SOFA_HAVE_EIGEN_UNSUPPORTED_AND_SUPERLU
#define EIGEN_SUPERLU_SUPPORT
#include <Eigen/SuperLUSupport>
#endif
#ifdef SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD
#include <unsupported/Eigen/CholmodSupport>
#endif
#include <Eigen/Sparse>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define EigenSparseSquareMatrix_CHECK
//#define EigenSparseSquareMatrix_VERBOSE


/** Container of a square Eigen::DynamicSparseMatrix matrix, to perform computations on DataTypes::VecDeriv vectors.
  The vectors are converted to/from Eigen format during the computations. The main interest of this class is to easily use efficient linear solvers.
  Linear equations system can be solved using LDLt or LU decomposition if you have installed the necessary extensions:

  For LDLT, you need the unsupported modules of Eigen, and cholmod. On Ubuntu 11.10, cholmod is included in the sparsesuite-dev package.

  For LU, you need the unsupported modules of Eigen, and superlu. On Ubuntu 11.10,, cholmod is included in the superlu3-dev package.

  These solvers require that you first decompose the system matrix using method <something>Decompose.
  Then, method <something>Solve(VecDeriv& x, const VecDeriv& b) can be used an arbitrary number of times.

  There is no checking that the matrix is actually square, but the solvers make no sense for non-square matrices.
  If you need a rectangular matrix, use EigenSparseRectangularMatrix
  */
template<typename DataTypes>
class EigenSparseSquareMatrix : public defaulttype::BaseMatrix
{

protected:
    typedef typename DataTypes::Real Real;
    typedef Eigen::DynamicSparseMatrix<Real> Matrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;

    Matrix eigenMatrix;    ///< the data

public:
    typedef typename DataTypes::VecDeriv VecDeriv;


    EigenSparseSquareMatrix(int nbRow=0, int nbCol=0)
    {
        resize(nbRow,nbCol);
    }

    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(int nbRow, int nbCol)
    {
        eigenMatrix.resize(nbRow,nbCol);
    }



    /// compute result = A * data
    void mult( VecDeriv& result, const VecDeriv& data ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(rowSize(),1),aux2(rowSize(),1);
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                aux1[DataTypes::deriv_total_size* i+j] = data[i][j];
        }
        // compute the product
        aux2 = eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                result[i][j] = aux2[DataTypes::deriv_total_size* i+j];
        }
    }

    /// compute result += A * data
    void addMult( VecDeriv& result, const VecDeriv& data ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(rowSize()),aux2(rowSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                aux1[DataTypes::deriv_total_size* i+j] = data[i][j];
        }
        // compute the product
        aux2 = eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                result[i][j] += aux2[DataTypes::deriv_total_size* i+j];
        }
    }

    /// number of rows
    unsigned int rowSize(void) const
    {
        return eigenMatrix.rows();
    }

    /// number of columns
    unsigned int colSize(void) const
    {
        return eigenMatrix.cols();
    }

    SReal element(int i, int j) const
    {
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        return eigenMatrix.coeff(i,j);
    }

    void set(int i, int j, double v)
    {
#ifdef EigenSparseSquareMatrix_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenMatrix.coeffRef(i,j) = (Real)v;
    }

    void add(int i, int j, double v)
    {
#ifdef EigenSparseSquareMatrix_VERBOSE
        std::cout << /*this->Name() << */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v<<std::endl;
#endif
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "/*<<this->Name()*/<<" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenMatrix.coeffRef(i,j) += (Real)v;
    }

    void clear(int i, int j)
    {
#ifdef EigenSparseSquareMatrix_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0"<<std::endl;
#endif
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenMatrix.coeffRef(i,j) = (Real)0;
    }

    ///< Set all the entries of a row to 0. Not efficient !
    void clearRow(int i)
    {
#ifdef EigenSparseSquareMatrix_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0"<<std::endl;
#endif
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize())
        {
            std::cerr << "ERROR: invalid write access to row "<<i<<" in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        for (unsigned j=0; j<colSize(); ++j)
            if(eigenMatrix.coeff(i,j)!=0)  eigenMatrix.coeffRef(i,j) = (Real)0;  ///< Not efficient !
    }

    ///< Set all the entries of a column to 0. Not efficient !
    void clearCol(int j)
    {
#ifdef EigenSparseSquareMatrix_VERBOSE
        std::cout <</* this->Name() << */"("<<rowSize()<<","<<colSize()<<"): col("<<j<<") = 0"<<std::endl;
#endif
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to column "<<j<<" in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        for (unsigned i=0; i<rowSize(); ++i)
            if(eigenMatrix.coeff(i,j)!=0)  eigenMatrix.coeffRef(i,j) = (Real)0;  ///< Not efficient !
    }

    ///< Set all the entries of a column and a row to 0. Not efficient !
    void clearRowCol(int i)
    {
#ifdef EigenSparseSquareMatrix_VERBOSE
        std::cout << /*this->Name() << */"("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0 and col("<<i<<") = 0"<<std::endl;
#endif
#ifdef EigenSparseSquareMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)i >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to row and column "<<i<<" in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        clearRow(i);  ///< Not efficient !
        clearCol(i);
    }

    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear()
    {
        resize(0,0);
        resize(rowSize(),colSize());
    }

    /// Matrix-vector product
    void mult( VectorEigen& result, const VectorEigen& data )
    {
        result = eigenMatrix * data;
    }


    friend std::ostream& operator << (std::ostream& out, const EigenSparseSquareMatrix<DataTypes>& v )
    {
        int nx = v.colSize();
        int ny = v.rowSize();
        out << "[";
        for (int y=0; y<ny; ++y)
        {
            out << "\n[";
            for (int x=0; x<nx; ++x)
            {
                out << " " << v.element(y,x);
            }
            out << " ]";
        }
        out << " ]";
        return out;
    }

    static const char* Name();

#ifdef SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD
protected:
    // sparse LDLT support
    typedef Eigen::SparseLDLT<Eigen::SparseMatrix<Real>,Eigen::Cholmod>  SparseLDLT;  // process SparseMatrix, not DynamicSparseMatrix (not implemented in Cholmod)
    SparseLDLT sparseLDLT; ///< used to factorize the matrix and solve systems using Cholesky method, for symmetric positive definite matrices only.
public:
    /// Try to compute the LDLT decomposition, and return true if success. The matrix is unchanged.
    bool ldltDecompose()
    {
        Eigen::SparseMatrix<Real> sparse(eigenMatrix); // DynamicSparse -> Sparse because LDLT not implemented on DynamicSparse
        sparseLDLT.compute(sparse);
        if( !sparseLDLT.succeeded() )
        {
            std::cerr<<"EigenSparseSquareMatrix::factorize() failed" << std::endl;
            return false;
        }
        return true;
    }

    /// Solve Ax=b, where A is this matrix. WARNING: ldltDecompose() must be called first. x and b can be the same vector.
    void ldltSolve( VecDeriv& x, const VecDeriv& b ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(rowSize(),1);
        for(unsigned i=0; i<b.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                aux1[DataTypes::deriv_total_size* i+j] = b[i][j];
        }
        // solve the equation
        sparseLDLT.solveInPlace(aux1);
        // convert the result back to the Sofa type
        for(unsigned i=0; i<x.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                x[i][j] = aux1[DataTypes::deriv_total_size* i+j];
        }
    }
#endif


#ifdef SOFA_HAVE_EIGEN_UNSUPPORTED_AND_SUPERLU
protected:
    // sparse LU support
    typedef Eigen::SparseLU<Eigen::SparseMatrix<Real>,Eigen::SuperLU >  SparseLU;      // process SparseMatrix, not DynamicSparseMatrix
    SparseLU sparseLU;     ///< used to factorize the matrix and solve systems using LU decomposition
public:
    /// Try to compute the LU decomposition, and return true if success. The matrix is unchanged.
    bool luDecompose()
    {
        Eigen::SparseMatrix<Real> sparse(eigenMatrix); // DynamicSparse -> Sparse because LDLT not implemented on DynamicSparse
        sparseLU.compute(sparse);
        if( !sparseLU.succeeded() )
        {
            std::cerr<<"EigenSparseSquareMatrix::luDecompose() failed" << std::endl;
            return false;
        }
        return true;
    }

    /// Solve Ax=b, where A is this matrix. WARNING: luDecompose() must be called first. x and b can be the same vector.
    void luSolve( VecDeriv& x, const VecDeriv& b ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(rowSize(),1),aux2(rowSize(),1);
        for(unsigned i=0; i<b.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                aux1[DataTypes::deriv_total_size* i+j] = b[i][j];
        }
        // solve the equation
        sparseLU.solve(aux1,&aux2);
        // convert the result back to the Sofa type
        for(unsigned i=0; i<x.size(); i++)
        {
            for(unsigned j=0; j<DataTypes::deriv_total_size; j++)
                x[i][j] = aux2[DataTypes::deriv_total_size* i+j];
        }
//        std::cerr<<"luSolve, b = " << aux1.transpose() << std::endl;
//        std::cerr<<"luSolve, x = " << aux2.transpose() << std::endl;
//        std::cerr<<"luSolve, check result Ax-b = " << (eigenMatrix * aux2 - aux1).transpose() << std::endl;
    }
#endif



};

template<> inline const char* EigenSparseSquareMatrix<defaulttype::Vec3dTypes >::Name() { return "EigenSparseSquareMatrix3d"; }
template<> inline const char* EigenSparseSquareMatrix<defaulttype::Vec3fTypes >::Name() { return "EigenSparseSquareMatrix3f"; }





} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
