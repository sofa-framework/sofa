/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#pragma once
#include <Eigen/Sparse>

namespace Eigen
{

template<typename Lhs, typename Rhs>        class SparseTimeDenseProduct_MT;
template<typename Lhs, typename Rhs, int InnerSize = internal::traits<Lhs>::ColsAtCompileTime> struct SparseDenseProductReturnType_MT;

template<typename Lhs, typename Rhs, int InnerSize> struct SparseDenseProductReturnType_MT
{
  typedef SparseTimeDenseProduct_MT<Lhs,Rhs> Type;
};

template<typename Lhs, typename Rhs> struct SparseDenseProductReturnType_MT<Lhs,Rhs,1>
{
    typedef SparseDenseOuterProduct<Lhs,Rhs,false> Type;
};

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
