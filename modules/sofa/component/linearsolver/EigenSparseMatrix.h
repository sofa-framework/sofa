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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenSparseMatrix_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenSparseMatrix_H

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <Eigen/Sparse>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define EigenSparseMatrix_CHECK
//#define EigenSparseMatrix_VERBOSE


/** Container of an Eigen::SparseMatrix<Real, RowMajor> matrix, able to perform computations on InDataTypes::VecDeriv and OutDataTypes::VecDeriv vectors.
  The vectors are converted to/from Eigen format during the computations.

  WARNING: Random write is not possible. For efficiency, the filling must be performed per row, column in increasing order.
Method beginRow(int index) must be called before any entry can be appended to row i.
Then set(i,j,value) must be used in for increasing j. There is no need to explicitly end a row.
When all the entries are written, method endEdit() must be applied to finalize the matrix.
  */
template<class InDataTypes, class OutDataTypes>
class EigenSparseMatrix : public defaulttype::BaseMatrix
{
public:

    typedef typename InDataTypes::Real Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;

    Matrix eigenMatrix;    ///< the data

    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
    typedef defaulttype::Mat<Nout,Nin,Real> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.


    EigenSparseMatrix(int nbRow=0, int nbCol=0)
    {
        resize(nbRow,nbCol);
    }

    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(int nbRow, int nbCol)
    {
        eigenMatrix.resize(nbRow,nbCol);
    }

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlockRows, int nbBlockCols)
    {
        eigenMatrix.resize(nbBlockRows * Nout, nbBlockCols * Nin);
    }

    void endEdit()
    {
        eigenMatrix.finalize();
    }



    /// compute result = A * data
    void mult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(colSize(),1), aux2(rowSize(),1);
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] = aux2[Nout* i+j];
        }
    }

    /// compute result = A * data
    void mult( Data<OutVecDeriv>& _result, const Data<InVecDeriv>& _data ) const
    {
        helper::WriteAccessor<Data<OutVecDeriv> > result (_result);
        helper::ReadAccessor<Data<InVecDeriv> > data (_data);

        // convert the data to Eigen type
        VectorEigen aux1(colSize(),1), aux2(rowSize(),1);
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] = aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(colSize()),aux2(rowSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] += aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const
    {
        // convert the data to Eigen type
        VectorEigen aux1(rowSize()),aux2(colSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                aux1[Nout* i+j] = data[i][j];
        }
        // compute the product
        aux2 = eigenMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                result[i][j] += aux2[Nin* i+j];
        }
    }
    /// compute result += A * data
    void addMultTranspose( Data<InVecDeriv>& result, const Data<OutVecDeriv>& data ) const
    {
        helper::WriteAccessor<Data<InVecDeriv> > res (result);
        helper::ReadAccessor<Data<OutVecDeriv> > dat (data);
        // convert the data to Eigen type
        VectorEigen aux1(rowSize()),aux2(colSize());
        for(unsigned i=0; i<dat.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                aux1[Nout* i+j] = dat[i][j];
        }
        // compute the product
        aux2 = eigenMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                res[i][j] += aux2[Nin* i+j];
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
#ifdef EigenSparseMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        return eigenMatrix.coeff(i,j);
    }

    /// must be called before inserting any element in the given row
    void beginRow( int i )
    {
        eigenMatrix.startVec(i);
    }

    /// This is efficient only if done in storing order: line, row
    void set(int i, int j, double v)
    {
#ifdef EigenSparseMatrix_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef EigenSparseMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenMatrix.insertBack(i,j) = (Real)v;
//        cerr<<"EigenSparseMatrix::set, size = "<< eigenMatrix.rows()<<", "<< eigenMatrix.cols()<<", entry: "<< i <<", "<<j<<" = "<< v << endl;
    }





    void add(int /*i*/, int /*j*/, double /*v*/)
    {
        cerr<<"EigenSparseMatrix::add(int i, int j, double v) is not implemented !"<<endl;
    }

    void clear(int i, int j)
    {
#ifdef EigenSparseMatrix_VERBOSE
        std::cout << /*this->Name() <<*/ "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0"<<std::endl;
#endif
#ifdef EigenSparseMatrix_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</*this->Name()<<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        eigenMatrix.coeffRef(i,j) = (Real)0;
    }

    ///< Set all the entries of a row to 0. Not efficient !
    void clearRow(int /*i*/)
    {
        cerr<<"EigenSparseMatrix::clearRow(int i) is not implemented !"<<endl;
    }

    ///< Set all the entries of a column to 0. Not efficient !
    void clearCol(int /*j*/)
    {
        cerr<<"EigenSparseMatrix::clearCol(int i) is not implemented !"<<endl;
    }

    ///< Set all the entries of a column and a row to 0. Not efficient !
    void clearRowCol(int /*i*/)
    {
        cerr<<"EigenSparseMatrix::clearRowCol(int i) is not implemented !"<<endl;
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


    friend std::ostream& operator << (std::ostream& out, const EigenSparseMatrix<InDataTypes,OutDataTypes>& v )
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


};

template<> inline const char* EigenSparseMatrix<defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >::Name() { return "EigenSparseMatrix3_1d"; }
template<> inline const char* EigenSparseMatrix<defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >::Name() { return "EigenSparseMatrix3_1f"; }





} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
