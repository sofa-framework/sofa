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

/// try to add opemp instructions to parallelize the eigen sparse matrix multiplication
/// inspired by eigen3.2.0 ConservativeSparseSparseProduct.h & SparseProduct.h
/// @warning this will not work with pruning (cf Eigen doc)
/// @warning this is based on the implementation of eigen3.2.0, it may be wrong with other versions

#ifndef EIGENBASESPARSEMATRIX_MT_H
#define EIGENBASESPARSEMATRIX_MT_H

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<typename Lhs, typename Rhs, typename ResultType>
static void conservative_sparse_sparse_product_MT(const Lhs& lhs, const Rhs& rhs, ResultType& res)
{
  typedef typename Eigen::internal::remove_all<Lhs>::type::Scalar Scalar;
  typedef typename Eigen::internal::remove_all<Lhs>::type::Index Index;

  // make sure to call innerSize/outerSize since we fake the storage order.
  const Index rows = lhs.innerSize();
  const Index cols = rhs.outerSize();
  eigen_assert(lhs.outerSize() == rhs.innerSize());

//  Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> mask(cols,rows);
  Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> values(cols,rows);
//  Eigen::Matrix<Index,Eigen::Dynamic,Eigen::Dynamic>  indices(cols,rows);
//  std::vector< std::vector<Index> > indices(cols);
  values.setZero();

  std::vector<Index> nnz(cols,0);


Index total = 0;

  // we compute each column of the result, one after the other
#pragma omp parallel for /*num_threads(8)*/ shared(values,nnz/*,indices*/) reduction(+:total)
  for (Index j=0; j<cols; ++j)
  {
//      printf("Element %d traité par le thread %d \n",j,omp_get_thread_num());

//      indices[j].reserve(rows);

    for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
    {
      Scalar y = rhsIt.value();
      Index k = rhsIt.index();
      for (typename Lhs::InnerIterator lhsIt(lhs, k); lhsIt; ++lhsIt)
      {
        Index i = lhsIt.index();
        Scalar x = lhsIt.value();
//        if(!values(j,i))
//        {
////          mask(j,i) = true;
//          values(j,i) = x * y;
////          indices(j,nnz[j])=i;
//          indices[j].push_back(i);
////          ++nnz[j];
//        }
//        else
          values(j,i) += x * y;
      }
    }

//    std::sort( indices[j].begin(), indices[j].end() );


    for(Index i=0; i<rows; ++i)
    {
        if( values(j,i)!=0 ) nnz[j]++;
    }
    total += nnz[j];
  }

  // estimate the number of non zero entries
  // given a rhs column containing Y non zeros, we assume that the respective Y columns
  // of the lhs differs in average of one non zeros, thus the number of non zeros for
  // the product of a rhs column with the lhs is X+Y where X is the average number of non zero
  // per column of the lhs.
  // Therefore, we have nnz(lhs*rhs) = nnz(lhs) + nnz(rhs)
//  Index estimated_nnz_prod = lhs.nonZeros() + rhs.nonZeros();

//  res.setZero();
//  res.reserve(Index(estimated_nnz_prod));


  // TODO eigen black magic to fill the matrix in parallel
  // start index for each colum (not parallel)
  // fill array structure in parallel
//  for (Index j=0; j<cols; ++j)
//  {
//     res.startVec(j);

//    // unordered insertion
////     for(int k=0; k</*nnz[j]*/indices[j].size(); ++k)
////    {
////      int i = indices[j][k];
////      res.insertBackByOuterInnerUnordered(j,i) = values(j,i);
////    }

//     // ordered insertion
////     for(unsigned int k=0; k<indices[j].size(); ++k)
////     {
////       int i = indices[j][k];
////       res.insertBack(j,i) = values(j,i);
////     }

//     for(Index i=0; i<rows; ++i)
//          {
//         if( values(j,i)!=0 )
//            res.insertBack(j,i) = values(j,i);
//          }
//  }

  //  res.finalize();



  // this is where eigen black magic occurs

  res.makeCompressed();
  res.reserve(total);

  Index* innerIndex = res.innerIndexPtr();
  Scalar* value = res.valuePtr();
  Index* outerIndex = res.outerIndexPtr(); outerIndex[cols]=total;



#pragma omp parallel for shared(values,nnz)
  for (Index j=0; j<cols; ++j)
  {
      outerIndex[j] = 0;
      for (Index k=0; k<j; ++k)
          outerIndex[j] += nnz[k];

        unsigned localIndex = 0;
        for(Index i=0; i<rows; ++i)
        {
            if( values(j,i)!=0 )
            {
                innerIndex[outerIndex[j]+localIndex] = i;
                value[outerIndex[j]+localIndex] = values(j,i);
                ++localIndex;
            }
        }
  }


}

static const unsigned int RowMajor = Eigen::RowMajor;
static const unsigned int ColMajor = Eigen::ColMajor;

template<typename Lhs, typename Rhs, typename ResultType,
  int LhsStorageOrder = (Eigen::internal::traits<Lhs>::Flags&Eigen::RowMajorBit) ? RowMajor : ColMajor,
  int RhsStorageOrder = (Eigen::internal::traits<Rhs>::Flags&Eigen::RowMajorBit) ? RowMajor : ColMajor,
  int ResStorageOrder = (Eigen::internal::traits<ResultType>::Flags&Eigen::RowMajorBit) ? RowMajor : ColMajor>
struct conservative_sparse_sparse_product_selector_MT;

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,ColMajor,ColMajor,ColMajor>
{
  typedef typename Eigen::internal::remove_all<Lhs>::type LhsCleaned;
  typedef typename LhsCleaned::Scalar Scalar;

  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,RowMajor> RowMajorMatrix;
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,ColMajor> ColMajorMatrix;
    ColMajorMatrix resCol(lhs.rows(),rhs.cols());
    conservative_sparse_sparse_product_MT<Lhs,Rhs,ColMajorMatrix>(lhs, rhs, resCol);
    // sort the non zeros:
    RowMajorMatrix resRow(resCol);
    res = resRow;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,RowMajor,ColMajor,ColMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
     typedef Eigen::SparseMatrix<typename ResultType::Scalar,RowMajor> RowMajorMatrix;
     RowMajorMatrix rhsRow = rhs;
     RowMajorMatrix resRow(lhs.rows(), rhs.cols());
     conservative_sparse_sparse_product_MT<RowMajorMatrix,Lhs,RowMajorMatrix>(rhsRow, lhs, resRow);
     res = resRow;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,ColMajor,RowMajor,ColMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,RowMajor> RowMajorMatrix;
    RowMajorMatrix lhsRow = lhs;
    RowMajorMatrix resRow(lhs.rows(), rhs.cols());
    conservative_sparse_sparse_product_MT<Rhs,RowMajorMatrix,RowMajorMatrix>(rhs, lhsRow, resRow);
    res = resRow;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,RowMajor,RowMajor,ColMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,RowMajor> RowMajorMatrix;
    RowMajorMatrix resRow(lhs.rows(), rhs.cols());
    conservative_sparse_sparse_product_MT<Rhs,Lhs,RowMajorMatrix>(rhs, lhs, resRow);
    res = resRow;
  }
};


template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,ColMajor,ColMajor,RowMajor>
{
  typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<Lhs>::type>::Scalar Scalar;

  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,ColMajor> ColMajorMatrix;
    ColMajorMatrix resCol(lhs.rows(), rhs.cols());
    conservative_sparse_sparse_product_MT<Lhs,Rhs,ColMajorMatrix>(lhs, rhs, resCol);
    res = resCol;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,RowMajor,ColMajor,RowMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,ColMajor> ColMajorMatrix;
    ColMajorMatrix lhsCol = lhs;
    ColMajorMatrix resCol(lhs.rows(), rhs.cols());
    conservative_sparse_sparse_product_MT<ColMajorMatrix,Rhs,ColMajorMatrix>(lhsCol, rhs, resCol);
    res = resCol;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,ColMajor,RowMajor,RowMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,ColMajor> ColMajorMatrix;
    ColMajorMatrix rhsCol = rhs;
    ColMajorMatrix resCol(lhs.rows(), rhs.cols());
    conservative_sparse_sparse_product_MT<Lhs,ColMajorMatrix,ColMajorMatrix>(lhs, rhsCol, resCol);
    res = resCol;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector_MT<Lhs,Rhs,ResultType,RowMajor,RowMajor,RowMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,RowMajor> RowMajorMatrix;
    typedef Eigen::SparseMatrix<typename ResultType::Scalar,ColMajor> ColMajorMatrix;
    RowMajorMatrix resRow(lhs.rows(),rhs.cols());
    conservative_sparse_sparse_product_MT<Rhs,Lhs,RowMajorMatrix>(rhs, lhs, resRow);
    // sort the non zeros:
    ColMajorMatrix resCol(resRow);
    res = resCol;
  }
};

/// Eigen::SparseMatrix multiplication (openmp multithreaded version)
/// @warning res MUST NOT be the same variable as lhs or rhs
template<typename Lhs, typename Rhs, typename ResultType>
void mul_EigenSparseMatrix_MT( ResultType& res, const Lhs& lhs, const Rhs& rhs )
{
#ifdef _OPENMP
    assert( &res != &lhs );
    assert( &res != &rhs );
    conservative_sparse_sparse_product_selector_MT< Lhs, Rhs, ResultType >::run(lhs, rhs, res);
#else
    res = lhs * rhs;
#endif
}


}
}
}



/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// SPARSE * DENSE
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

#include <Eigen/Sparse>

namespace Eigen
{

template<typename Lhs, typename Rhs>        class SparseTimeDenseProduct_MT;
//template<typename Lhs, typename Rhs>        class DenseTimeSparseProduct_MT;
//template<typename Lhs, typename Rhs, bool Transpose> class SparseDenseOuterProduct_MT;
//template<typename Lhs, typename Rhs, int InnerSize = internal::traits<Lhs>::ColsAtCompileTime> struct DenseSparseProductReturnType;
template<typename Lhs, typename Rhs, int InnerSize = internal::traits<Lhs>::ColsAtCompileTime> struct SparseDenseProductReturnType_MT;

template<typename Lhs, typename Rhs, int InnerSize> struct SparseDenseProductReturnType_MT
{
  typedef SparseTimeDenseProduct_MT<Lhs,Rhs> Type;
};

template<typename Lhs, typename Rhs> struct SparseDenseProductReturnType_MT<Lhs,Rhs,1>
{
//  typedef SparseDenseOuterProduct_MT<Lhs,Rhs,false> Type;
    typedef SparseDenseOuterProduct<Lhs,Rhs,false> Type;
};

//template<typename Lhs, typename Rhs, int InnerSize> struct DenseSparseProductReturnType_MT
//{
//  typedef DenseTimeSparseProduct_MT<Lhs,Rhs> Type;
//};

//template<typename Lhs, typename Rhs> struct DenseSparseProductReturnType_MT<Lhs,Rhs,1>
//{
//  typedef SparseDenseOuterProduct_MT<Rhs,Lhs,true> Type;
//};


//namespace internal {

//using Eigen::internal::traits;

//template<typename Lhs, typename Rhs, bool Tr>
//struct traits<SparseDenseOuterProduct_MT<Lhs,Rhs,Tr> >
//{
//  typedef Sparse StorageKind;
//  typedef typename scalar_product_traits<typename traits<Lhs>::Scalar,
//                                         typename traits<Rhs>::Scalar>::ReturnType Scalar;
//  typedef typename Lhs::Index Index;
//  typedef typename Lhs::Nested LhsNested;
//  typedef typename Rhs::Nested RhsNested;
//  typedef typename remove_all<LhsNested>::type _LhsNested;
//  typedef typename remove_all<RhsNested>::type _RhsNested;

//  enum {
//    LhsCoeffReadCost = traits<_LhsNested>::CoeffReadCost,
//    RhsCoeffReadCost = traits<_RhsNested>::CoeffReadCost,

//    RowsAtCompileTime    = Tr ? int(traits<Rhs>::RowsAtCompileTime)     : int(traits<Lhs>::RowsAtCompileTime),
//    ColsAtCompileTime    = Tr ? int(traits<Lhs>::ColsAtCompileTime)     : int(traits<Rhs>::ColsAtCompileTime),
//    MaxRowsAtCompileTime = Tr ? int(traits<Rhs>::MaxRowsAtCompileTime)  : int(traits<Lhs>::MaxRowsAtCompileTime),
//    MaxColsAtCompileTime = Tr ? int(traits<Lhs>::MaxColsAtCompileTime)  : int(traits<Rhs>::MaxColsAtCompileTime),

//    Flags = Tr ? RowMajorBit : 0,

//    CoeffReadCost = LhsCoeffReadCost + RhsCoeffReadCost + NumTraits<Scalar>::MulCost
//  };
//};

//} // end namespace internal

//template<typename Lhs, typename Rhs, bool Tr>
//class SparseDenseOuterProduct_MT
// : public SparseMatrixBase<SparseDenseOuterProduct_MT<Lhs,Rhs,Tr> >
//{
//  public:

//    typedef SparseMatrixBase<SparseDenseOuterProduct_MT> Base;
//    EIGEN_DENSE_PUBLIC_INTERFACE(SparseDenseOuterProduct_MT)
//    typedef internal::traits<SparseDenseOuterProduct_MT> Traits;

//  private:

//    typedef typename Traits::LhsNested LhsNested;
//    typedef typename Traits::RhsNested RhsNested;
//    typedef typename Traits::_LhsNested _LhsNested;
//    typedef typename Traits::_RhsNested _RhsNested;

//  public:

//    class InnerIterator;

//    EIGEN_STRONG_INLINE SparseDenseOuterProduct_MT(const Lhs& lhs, const Rhs& rhs)
//      : m_lhs(lhs), m_rhs(rhs)
//    {
//      EIGEN_STATIC_ASSERT(!Tr,YOU_MADE_A_PROGRAMMING_MISTAKE);
//    }

//    EIGEN_STRONG_INLINE SparseDenseOuterProduct_MT(const Rhs& rhs, const Lhs& lhs)
//      : m_lhs(lhs), m_rhs(rhs)
//    {
//      EIGEN_STATIC_ASSERT(Tr,YOU_MADE_A_PROGRAMMING_MISTAKE);
//    }

//    EIGEN_STRONG_INLINE Index rows() const { return Tr ? m_rhs.rows() : m_lhs.rows(); }
//    EIGEN_STRONG_INLINE Index cols() const { return Tr ? m_lhs.cols() : m_rhs.cols(); }

//    EIGEN_STRONG_INLINE const _LhsNested& lhs() const { return m_lhs; }
//    EIGEN_STRONG_INLINE const _RhsNested& rhs() const { return m_rhs; }

//  protected:
//    LhsNested m_lhs;
//    RhsNested m_rhs;
//};

//template<typename Lhs, typename Rhs, bool Transpose>
//class SparseDenseOuterProduct_MT<Lhs,Rhs,Transpose>::InnerIterator : public _LhsNested::InnerIterator
//{
//    typedef typename _LhsNested::InnerIterator Base;
//    typedef typename SparseDenseOuterProduct_MT::Index Index;
//  public:
//    EIGEN_STRONG_INLINE InnerIterator(const SparseDenseOuterProduct_MT& prod, Index outer)
//      : Base(prod.lhs(), 0), m_outer(outer), m_factor(prod.rhs().coeff(outer))
//    {
//    }

//    inline Index outer() const { return m_outer; }
//    inline Index row() const { return Transpose ? Base::row() : m_outer; }
//    inline Index col() const { return Transpose ? m_outer : Base::row(); }

//    inline Scalar value() const { return Base::value() * m_factor; }

//  protected:
//    int m_outer;
//    Scalar m_factor;
//};

namespace internal {
template<typename Lhs, typename Rhs>
struct traits<SparseTimeDenseProduct_MT<Lhs,Rhs> >
 : traits<ProductBase<SparseTimeDenseProduct_MT<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
  typedef MatrixXpr XprKind;
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType,
         int LhsStorageOrder = ((SparseLhsType::Flags&RowMajorBit)==RowMajorBit) ? RowMajor : ColMajor,
         bool ColPerCol = ((DenseRhsType::Flags&RowMajorBit)==0) || DenseRhsType::ColsAtCompileTime==1>
struct sparse_time_dense_product_impl_MT;

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl_MT<SparseLhsType,DenseRhsType,DenseResType, RowMajor, true>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::Index Index;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha, unsigned nbThreads)
  {
#pragma omp parallel for num_threads(nbThreads) //schedule(static,3000)
      for(Index j=0; j<lhs.outerSize(); ++j)
      {
          for(Index c=0; c<rhs.cols(); ++c)
          {
            typename Res::Scalar& r = res.coeffRef(j,c);
            r = 0;
            for(LhsInnerIterator it(lhs,j); it ;++it)
                r += it.value() * rhs.coeff(it.index(),c);
            r *= alpha;
          }
      }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl_MT<SparseLhsType,DenseRhsType,DenseResType, ColMajor, true>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  typedef typename Lhs::Index Index;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha, unsigned nbThreads)
  {
#pragma omp parallel for num_threads(nbThreads)
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      for(Index c=0; c<rhs.cols(); ++c)
      {
        typename Res::Scalar rhs_j = alpha * rhs.coeff(j,c);
        for(LhsInnerIterator it(lhs,j); it ;++it)
          res.coeffRef(it.index(),c) += it.value() * rhs_j;
      }
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl_MT<SparseLhsType,DenseRhsType,DenseResType, RowMajor, false>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  typedef typename Lhs::Index Index;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha, unsigned nbThreads)
  {
#pragma omp parallel for num_threads(nbThreads)
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      typename Res::RowXpr res_j(res.row(j));
      for(LhsInnerIterator it(lhs,j); it ;++it)
        res_j += (alpha*it.value()) * rhs.row(it.index());
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl_MT<SparseLhsType,DenseRhsType,DenseResType, ColMajor, false>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::InnerIterator LhsInnerIterator;
  typedef typename Lhs::Index Index;
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha, unsigned nbThreads)
  {
    #pragma omp parallel for num_threads(nbThreads)
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      typename Rhs::ConstRowXpr rhs_j(rhs.row(j));
      for(LhsInnerIterator it(lhs,j); it ;++it)
        res.row(it.index()) += (alpha*it.value()) * rhs_j;
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType,typename AlphaType>
inline void sparse_time_dense_product_MT(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha, unsigned nbThreads)
{
  sparse_time_dense_product_impl_MT<SparseLhsType,DenseRhsType,DenseResType>::run(lhs, rhs, res, alpha, nbThreads);
}

} // end namespace internal

template<typename Lhs, typename Rhs>
class SparseTimeDenseProduct_MT
  : public ProductBase<SparseTimeDenseProduct_MT<Lhs,Rhs>, Lhs, Rhs>
{
    unsigned m_nbThreads;
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(SparseTimeDenseProduct_MT)

    SparseTimeDenseProduct_MT(const Lhs& lhs, const Rhs& rhs, unsigned nbThreads) : Base(lhs,rhs), m_nbThreads(nbThreads)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
#ifdef _OPENMP
        // no multithreading for too small vectors
        if( ( m_rhs.cols() == 1 && m_rhs.rows()<3000 ) || m_nbThreads==1 )
            internal::sparse_time_dense_product(m_lhs, m_rhs, dest, alpha);
        else
        {
//            msg_info()<<"SparseTimeDenseProduct_MT: "<<m_nbThreads<<std::endl;
            internal::sparse_time_dense_product_MT<Lhs,Rhs,Dest,Scalar>(m_lhs, m_rhs, dest, alpha, m_nbThreads);
        }
#else
        internal::sparse_time_dense_product(m_lhs, m_rhs, dest, alpha);
#endif
    }

  private:
    SparseTimeDenseProduct_MT& operator=(const SparseTimeDenseProduct_MT&);
};



} // namespace Eigen


namespace sofa
{

namespace component
{

namespace linearsolver
{
#ifndef OMP_DEFAULT_NUM_THREADS_EIGEN_SPARSE_DENSE_PRODUCT
#define OMP_DEFAULT_NUM_THREADS_EIGEN_SPARSE_DENSE_PRODUCT 1
#endif

    /// Eigen::Sparse * Dense Matrices multiplication (openmp multi-threaded version)
    template<typename Derived, typename OtherDerived >
    inline const typename Eigen::SparseDenseProductReturnType_MT<Derived,OtherDerived>::Type
    mul_EigenSparseDenseMatrix_MT( const Eigen::SparseMatrixBase<Derived>& lhs, const Eigen::MatrixBase<OtherDerived>& rhs, unsigned nbThreads=OMP_DEFAULT_NUM_THREADS_EIGEN_SPARSE_DENSE_PRODUCT )
    {
        return typename Eigen::SparseDenseProductReturnType_MT<Derived,OtherDerived>::Type( lhs.derived(), rhs.derived(), nbThreads );
    }


}
}
}



#endif // EIGENBASESPARSEMATRIX_MT_H
