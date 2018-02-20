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
#ifndef SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <math.h>
#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Simple bloc full matrix container (used for InvMatrixType)
template<int N, typename T>
class BlocFullMatrix : public defaulttype::BaseMatrix
{
public:

    enum { BSIZE = N };
    typedef T Real;

    class TransposedBloc
    {
    public:
        const defaulttype::Mat<BSIZE,BSIZE,Real>& m;
        TransposedBloc(const defaulttype::Mat<BSIZE,BSIZE,Real>& m) : m(m) {}
        defaulttype::Vec<BSIZE,Real> operator*(const defaulttype::Vec<BSIZE,Real>& v)
        {
            return m.multTranspose(v);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            return -m.transposed();
        }
    };

    class Bloc : public defaulttype::Mat<BSIZE,BSIZE,Real>
    {
    public:
        Index Nrows() const { return BSIZE; }
        Index Ncols() const { return BSIZE; }
        void resize(Index, Index)
        {
            clear();
        }
        const T& element(Index i, Index j) const { return (*this)[i][j]; }
        void set(Index i, Index j, const T& v) { (*this)[i][j] = v; }
        void add(Index i, Index j, const T& v) { (*this)[i][j] += v; }
        void operator=(const defaulttype::Mat<BSIZE,BSIZE,Real>& v)
        {
            defaulttype::Mat<BSIZE,BSIZE,Real>::operator=(v);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator-();
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator-(const defaulttype::Mat<BSIZE,BSIZE,Real>& m) const
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator-(m);
        }
        defaulttype::Vec<BSIZE,Real> operator*(const defaulttype::Vec<BSIZE,Real>& v)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(v);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator*(const defaulttype::Mat<BSIZE,BSIZE,Real>& m)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(m);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator*(const Bloc& m)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(m);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator*(const TransposedBloc& mt)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(mt.m.transposed());
        }
        TransposedBloc t() const
        {
            return TransposedBloc(*this);
        }
        Bloc i() const
        {
            Bloc r;
            r.invert(*this);
            return r;
        }
    };
    typedef Bloc SubMatrixType;
    typedef FullMatrix<T> InvMatrixType;
    // return the dimension of submatrices when requesting a given size
    static Index getSubMatrixDim(Index) { return BSIZE; }

protected:
    Bloc* data;
    Index nTRow,nTCol;
    Index nBRow,nBCol;
    Index allocsize;

public:

    BlocFullMatrix()
        : data(NULL), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
    {
    }

    BlocFullMatrix(Index nbRow, Index nbCol)
        : data(new T[nbRow*nbCol]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow/BSIZE), nBCol(nbCol/BSIZE), allocsize((nbCol/BSIZE)*(nbRow/BSIZE))
    {
    }

    ~BlocFullMatrix()
    {
        if (allocsize>0)
            delete data;
    }

    Bloc* ptr() { return data; }
    const Bloc* ptr() const { return data; }

    const Bloc& bloc(Index bi, Index bj) const
    {
        return data[bi*nBCol + bj];
    }
    Bloc& bloc(Index bi, Index bj)
    {
        return data[bi*nBCol + bj];
    }

    void resize(Index nbRow, Index nbCol)
    {
        if (nbCol != nTCol || nbRow != nTRow)
        {
            if (allocsize < 0)
            {
                if ((nbCol/BSIZE)*(nbRow/BSIZE) > -allocsize)
                {
                    msg_error("BTDLinearSolver") << "Cannot resize preallocated matrix to size ("<<nbRow<<","<<nbCol<<")." ;
                    return;
                }
            }
            else
            {
                if ((nbCol/BSIZE)*(nbRow/BSIZE) > allocsize)
                {
                    if (allocsize > 0)
                        delete[] data;
                    allocsize = (nbCol/BSIZE)*(nbRow/BSIZE);
                    data = new Bloc[allocsize];
                }
            }
            nTCol = nbCol;
            nTRow = nbRow;
            nBCol = nbCol/BSIZE;
            nBRow = nbRow/BSIZE;
        }
        clear();
    }

    Index rowSize(void) const
    {
        return nTRow;
    }

    Index colSize(void) const
    {
        return nTCol;
    }

    SReal element(Index i, Index j) const
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        return bloc(bi,bj)[i][j];
    }

    const Bloc& asub(Index bi, Index bj, Index, Index) const
    {
        return bloc(bi,bj);
    }

    const Bloc& sub(Index i, Index j, Index, Index) const
    {
        return asub(i/BSIZE,j/BSIZE);
    }

    Bloc& asub(Index bi, Index bj, Index, Index)
    {
        return bloc(bi,bj);
    }

    Bloc& sub(Index i, Index j, Index, Index)
    {
        return asub(i/BSIZE,j/BSIZE);
    }

    template<class B>
    void getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
    {
        m = sub(i,j, nrow, ncol);
    }

    template<class B>
    void getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
    {
        m = asub(bi, bj, nrow, ncol);
    }

    template<class B>
    void setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
    {
        sub(i,j, nrow, ncol) = m;
    }

    template<class B>
    void setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
    {
        asub(bi, bj, nrow, ncol) = m;
    }

    void set(Index i, Index j, double v)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        bloc(bi,bj)[i][j] = (Real)v;
    }

    void add(Index i, Index j, double v)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        bloc(bi,bj)[i][j] += (Real)v;
    }

    void clear(Index i, Index j)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        bloc(bi,bj)[i][j] = (Real)0;
    }

    void clearRow(Index i)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        for (Index bj = 0; bj < nBCol; ++bj)
            for (Index j=0; j<BSIZE; ++j)
                bloc(bi,bj)[i][j] = (Real)0;
    }

    void clearCol(Index j)
    {
        Index bj = j / BSIZE; j = j % BSIZE;
        for (Index bi = 0; bi < nBRow; ++bi)
            for (Index i=0; i<BSIZE; ++i)
                bloc(bi,bj)[i][j] = (Real)0;
    }

    void clearRowCol(Index i)
    {
        clearRow(i);
        clearCol(i);
    }

    void clear()
    {
        for (Index i=0; i<3*nBRow; ++i)
            data[i].clear();
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res(rowSize());
        for (Index bi=0; bi<nBRow; ++bi)
        {
            Index bj = 0;
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index j=0; j<BSIZE; ++j)
                {
                    r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                }
                res[bi*BSIZE + i] = r;
            }
            for (++bj; bj<nBCol; ++bj)
            {
                for (Index i=0; i<BSIZE; ++i)
                {
                    Real r = 0;
                    for (Index j=0; j<BSIZE; ++j)
                    {
                        r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                    }
                    res[bi*BSIZE + i] += r;
                }
            }
        }
        return res;
    }


    static const char* Name();
};

template<int N, typename T>
class BlockVector : public FullVector<T>
{
public:
    typedef FullVector<T> Inherit;
    typedef T Real;
    typedef typename Inherit::Index Index;

    typedef typename Inherit::value_type value_type;
    typedef typename Inherit::size_type size_type;
    typedef typename Inherit::iterator iterator;
    typedef typename Inherit::const_iterator const_iterator;

    class Bloc : public defaulttype::Vec<N,T>
    {
    public:
        Index Nrows() const { return N; }
        void resize(Index) { this->clear(); }
        void operator=(const defaulttype::Vec<N,T>& v)
        {
            defaulttype::Vec<N,T>::operator=(v);
        }
        void operator=(int v)
        {
            defaulttype::Vec<N,T>::fill((float)v);
        }
        void operator=(float v)
        {
            defaulttype::Vec<N,T>::fill(v);
        }
        void operator=(double v)
        {
            defaulttype::Vec<N,T>::fill(v);
        }
    };

    typedef Bloc SubVectorType;

public:

    BlockVector()
    {
    }

    explicit BlockVector(Index n)
        : Inherit(n)
    {
    }

    virtual ~BlockVector()
    {
    }

    const Bloc& sub(Index i, Index) const
    {
        return (const Bloc&)*(this->ptr()+i);
    }

    Bloc& sub(Index i, Index)
    {
        return (Bloc&)*(this->ptr()+i);
    }

    const Bloc& asub(Index bi, Index) const
    {
        return (const Bloc&)*(this->ptr()+bi*N);
    }

    Bloc& asub(Index bi, Index)
    {
        return (Bloc&)*(this->ptr()+bi*N);
    }
};

/// Simple BTD matrix container
template<int N, typename T>
class BTDMatrix : public defaulttype::BaseMatrix
{
public:
    enum { BSIZE = N };
    typedef T Real;
    typedef typename defaulttype::BaseMatrix::Index Index;

    class TransposedBloc
    {
    public:
        const defaulttype::Mat<BSIZE,BSIZE,Real>& m;
        TransposedBloc(const defaulttype::Mat<BSIZE,BSIZE,Real>& m) : m(m) {}
        defaulttype::Vec<BSIZE,Real> operator*(const defaulttype::Vec<BSIZE,Real>& v)
        {
            return m.multTranspose(v);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            defaulttype::Mat<BSIZE,BSIZE,Real> r;
            for (Index i=0; i<BSIZE; i++)
                for (Index j=0; j<BSIZE; j++)
                    r[i][j]=-m[j][i];
            return r;
        }
    };

    class Bloc : public defaulttype::Mat<BSIZE,BSIZE,Real>
    {
    public:
        Index Nrows() const { return BSIZE; }
        Index Ncols() const { return BSIZE; }
        void resize(Index, Index)
        {
            this->clear();
        }
        const T& element(Index i, Index j) const { return (*this)[i][j]; }
        void set(Index i, Index j, const T& v) { (*this)[i][j] = v; }
        void add(Index i, Index j, const T& v) { (*this)[i][j] += v; }
        void operator=(const defaulttype::Mat<BSIZE,BSIZE,Real>& v)
        {
            defaulttype::Mat<BSIZE,BSIZE,Real>::operator=(v);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            defaulttype::Mat<BSIZE,BSIZE,Real> r;
            for (Index i=0; i<BSIZE; i++)
                for (Index j=0; j<BSIZE; j++)
                    r[i][j]=-(*this)[i][j];
            return r;
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator-(const defaulttype::Mat<BSIZE,BSIZE,Real>& m) const
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator-(m);
        }
        defaulttype::Vec<BSIZE,Real> operator*(const defaulttype::Vec<BSIZE,Real>& v)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(v);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator*(const defaulttype::Mat<BSIZE,BSIZE,Real>& m)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(m);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator*(const Bloc& m)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(m);
        }
        defaulttype::Mat<BSIZE,BSIZE,Real> operator*(const TransposedBloc& mt)
        {
            return defaulttype::Mat<BSIZE,BSIZE,Real>::operator*(mt.m.transposed());
        }
        TransposedBloc t() const
        {
            return TransposedBloc(*this);
        }
        Bloc i() const
        {
            Bloc r;
            r.invert(*this);
            return r;
        }
    };

    typedef Bloc SubMatrixType;
    typedef sofa::defaulttype::Mat<N,N,Real> BlocType;
    typedef BlocFullMatrix<N,T> InvMatrixType;
    // return the dimension of submatrices when requesting a given size
    static Index getSubMatrixDim(Index) { return BSIZE; }

protected:
    Bloc* data;
    Index nTRow,nTCol;
    Index nBRow,nBCol;
    Index allocsize;

public:

    BTDMatrix()
        : data(NULL), nTRow(0), nTCol(0), nBRow(0), nBCol(0), allocsize(0)
    {
    }

    BTDMatrix(Index nbRow, Index nbCol)
        : data(new T[3*(nbRow/BSIZE)]), nTRow(nbRow), nTCol(nbCol), nBRow(nbRow/BSIZE), nBCol(nbCol/BSIZE), allocsize(3*(nbRow/BSIZE))
    {
    }

    ~BTDMatrix()
    {
        if (allocsize>0)
            delete data;
    }

    Bloc* ptr() { return data; }
    const Bloc* ptr() const { return data; }

    //Real* operator[](Index i)
    //{
    //    return data+i*pitch;
    //}
    const Bloc& bloc(Index bi, Index bj) const
    {
        return data[3*bi + (bj - bi + 1)];
    }
    Bloc& bloc(Index bi, Index bj)
    {
        return data[3*bi + (bj - bi + 1)];
    }

    void resize(Index nbRow, Index nbCol)
    {
        if (nbCol != nTCol || nbRow != nTRow)
        {
            if (allocsize < 0)
            {
                if ((nbRow/BSIZE)*3 > -allocsize)
                {
                    msg_error("BTDLinearSolver") << "Cannot resize preallocated matrix to size ("<<nbRow<<","<<nbCol<<")" ;
                    return;
                }
            }
            else
            {
                if ((nbRow/BSIZE)*3 > allocsize)
                {
                    if (allocsize > 0)
                        delete[] data;
                    allocsize = (nbRow/BSIZE)*3;
                    data = new Bloc[allocsize];
                }
            }
            nTCol = nbCol;
            nTRow = nbRow;
            nBCol = nbCol/BSIZE;
            nBRow = nbRow/BSIZE;
        }
        clear();
    }

    Index rowSize(void) const
    {
        return nTRow;
    }

    Index colSize(void) const
    {
        return nTCol;
    }

    SReal element(Index i, Index j) const
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return (SReal)0;
        return data[bi*3+bindex][i][j];
    }

    const Bloc& asub(Index bi, Index bj, Index, Index) const
    {
        static Bloc b;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return b;
        return data[bi*3+bindex];
    }

    const Bloc& sub(Index i, Index j, Index, Index) const
    {
        return asub(i/BSIZE,j/BSIZE);
    }

    Bloc& asub(Index bi, Index bj, Index, Index)
    {
        static Bloc b;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return b;
        return data[bi*3+bindex];
    }

    Bloc& sub(Index i, Index j, Index, Index)
    {
        return asub(i/BSIZE,j/BSIZE);
    }

    template<class B>
    void getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m)
    {
        m = sub(i,j, nrow, ncol);
    }

    template<class B>
    void getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m)
    {
        m = asub(bi, bj, nrow, ncol);
    }

    template<class B>
    void setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m)
    {
        sub(i,j, nrow, ncol) = m;
    }

    template<class B>
    void setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m)
    {
        asub(bi, bj, nrow, ncol) = m;
    }

    void set(Index i, Index j, double v)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return;
        data[bi*3+bindex][i][j] = (Real)v;
    }

    void add(Index i, Index j, double v)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return;
        data[bi*3+bindex][i][j] += (Real)v;
    }

    void clear(Index i, Index j)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        Index bj = j / BSIZE; j = j % BSIZE;
        Index bindex = bj - bi + 1;
        if (bindex >= 3) return;
        data[bi*3+bindex][i][j] = (Real)0;
    }

    void clearRow(Index i)
    {
        Index bi = i / BSIZE; i = i % BSIZE;
        for (Index bj = 0; bj < 3; ++bj)
            for (Index j=0; j<BSIZE; ++j)
                data[bi*3+bj][i][j] = (Real)0;
    }

    void clearCol(Index j)
    {
        Index bj = j / BSIZE; j = j % BSIZE;
        if (bj > 0)
            for (Index i=0; i<BSIZE; ++i)
                data[(bj-1)*3+2][i][j] = (Real)0;
        for (Index i=0; i<BSIZE; ++i)
            data[bj*3+1][i][j] = (Real)0;
        if (bj < nBRow-1)
            for (Index i=0; i<BSIZE; ++i)
                data[(bj+1)*3+0][i][j] = (Real)0;
    }

    void clearRowCol(Index i)
    {
        clearRow(i);
        clearCol(i);
    }

    void clear()
    {
        for (Index i=0; i<3*nBRow; ++i)
            data[i].clear();
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res(rowSize());
        for (Index bi=0; bi<nBRow; ++bi)
        {
            Index b0 = (bi > 0) ? 0 : 1;
            Index b1 = ((bi < nBRow - 1) ? 3 : 2);
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index bj = b0; bj < b1; ++bj)
                {
                    for (Index j=0; j<BSIZE; ++j)
                    {
                        r += data[bi*3+bj][i][j] * v[(bi + bj - 1)*BSIZE + j];
                    }
                }
                res[bi*BSIZE + i] = r;
            }
        }
        return res;
    }

    static const char* Name();
};


/// Linear system solver using Thomas Algorithm for Block Tridiagonal matrices
///
/// References:
/// Conte, S.D., and deBoor, C. (1972). Elementary Numerical Analysis. McGraw-Hill, New York
/// http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
/// http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
/// http://www4.ncsu.edu/eos/users/w/white/www/white/ma580/chap2.5.PDF
template<class Matrix, class Vector>
class BTDLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<Matrix,Vector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BTDLinearSolver, Matrix, Vector), SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, Matrix, Vector));

    Data<bool> f_verbose; ///< Dump system state at each iteration
    Data<bool> problem; ///< display debug informations about subpartSolve computation
    Data<bool> subpartSolve; ///< Allows for the computation of a subpart of the system

    Data<bool> verification; ///< verification of the subpartSolve
    Data<bool> test_perf; ///< verification of performance

    typedef typename Vector::SubVectorType SubVector;
    typedef typename Matrix::SubMatrixType SubMatrix;
    typedef typename Vector::Real Real;
    typedef typename Matrix::BlocType BlocType;
    typedef typename Matrix::Index Index;
    typedef std::list<Index> ListIndex;
    typedef std::pair<Index,Index> IndexPair;
    typedef std::map<IndexPair, SubMatrix> MysparseM;
    typedef typename std::map<IndexPair, SubMatrix>::iterator MysparseMit;

    //helper::vector<SubMatrix> alpha;
    helper::vector<SubMatrix> alpha_inv;
    helper::vector<SubMatrix> lambda;
    helper::vector<SubMatrix> B;
    typename Matrix::InvMatrixType Minv;  //inverse matrix


    //////////////////////////// for subpartSolve
    MysparseM H; // force transfer
    MysparseMit H_it;
    Vector bwdContributionOnLH;  //
    Vector fwdContributionOnRH;

    Vector _rh_buf;		 //				// buf the right hand term
    //Vector _df_buf;		 //
    SubVector _acc_rh_bloc;		// accumulation of rh through the browsing of the structure
    SubVector _acc_lh_bloc;		// accumulation of lh through the browsing of the strucutre
    Index	current_bloc, first_block;
    std::vector<SubVector> Vec_dRH;			// buf the dRH on block that are not current_bloc...
    ////////////////////////////

    helper::vector<Index> nBlockComputedMinv;
    Vector Y;

    Data<int> f_blockSize; ///< dimension of the blocks in the matrix
protected:
    BTDLinearSolver()
        : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
        , problem(initData(&problem, false,"showProblem", "display debug informations about subpartSolve computation") )
        , subpartSolve(initData(&subpartSolve, false,"subpartSolve", "Allows for the computation of a subpart of the system") )
        , verification(initData(&verification, false,"verification", "verification of the subpartSolve"))
        , test_perf(initData(&test_perf, false,"test_perf", "verification of performance"))
        , f_blockSize( initData(&f_blockSize,6,"blockSize","dimension of the blocks in the matrix") )
    {
        Index bsize = Matrix::getSubMatrixDim(0);
        if (bsize > 0)
        {
            // the template uses fixed bloc size
            f_blockSize.setValue((int)bsize);
            f_blockSize.setReadOnly(true);
        }
    }
public:
    void my_identity(SubMatrix& Id, const Index size_id);

    void invert(SubMatrix& Inv, const BlocType& m);

    void invert(Matrix& M) override;

    void computeMinvBlock(Index i, Index j);

    double getMinvElement(Index i, Index j);

    /// Solve Mx=b
    void solve (Matrix& /*M*/, Vector& x, Vector& b) override;



    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override;


    /// Init the partial solve
    void init_partial_solve() override;

    using MatrixLinearSolver<Matrix,Vector>::partial_solve;
    /// partial solve :
    /// b is accumulated
    /// db is a sparse vector that is added to b
    /// partial_x is a sparse vector (with sparse map given) that provide the result of M x = b+db
    /// Solve Mx=b
    //void partial_solve_old(ListIndex&  Iout, ListIndex&  Iin , bool NewIn);
    void partial_solve(ListIndex&  Iout, ListIndex&  Iin , bool NewIn) override;



    void init_partial_inverse(const Index &nb, const Index &bsize);



    template<class RMatrix, class JMatrix>
    bool addJMInvJt(RMatrix& result, JMatrix& J, double fact);



private:


    Index _indMaxNonNullForce; // point with non null force which index is the greatest and for which globalAccumulate was not proceed

    Index _indMaxFwdLHComputed;  // indice of node from which bwdLH is accurate

    /// private functions for partial solve
    /// step1=> accumulate RH locally for the InBloc (only if a new force is detected on RH)
    void bwdAccumulateRHinBloc(Index indMaxBloc);   // indMaxBloc should be equal to _indMaxNonNullForce


    /// step2=> accumulate LH globally to step down the value of current_bloc to 0
    void bwdAccumulateLHGlobal( );


    /// step3=> accumulate RH globally to step up the value of current_bloc to the smallest value needed in OutBloc
    void fwdAccumulateRHGlobal(Index indMinBloc);


    /// step4=> compute solution for the indices in the bloc
    /// (and accumulate the potential local dRH (set in Vec_dRH) [set in step1] that have not been yet taken into account by the global bwd and fwd
    void fwdComputeLHinBloc(Index indMaxBloc);


};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_LINEAR_SOLVER_API BTDLinearSolver< BTDMatrix<6, double>, BlockVector<6, double> >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_LINEAR_SOLVER_API BTDLinearSolver< BTDMatrix<6, float>, BlockVector<6, float> >;
#endif


#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
