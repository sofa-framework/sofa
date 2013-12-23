
/// try to add opemp instructions to parallelize the eigen sparse matrix multiplication
/// inspired by eigen3.2.0 ConservativeSparseSparseProduct.h
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



} } }

#endif // EIGENBASESPARSEMATRIX_MT_H
