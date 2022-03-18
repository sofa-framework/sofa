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
* Authors: The SOFA Team && external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

/******************************************************************************
 * Initial author: Jean-Nicolas Brunet
 * From: https://github.com/jnbrunet/caribou
 ******************************************************************************/

#pragma once

#include <sofa/linearalgebra/config.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/eigen/EigenVector.h>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include <sofa/type/trait/suffix.h>

namespace sofa::linearalgebra::eigen
{

/**
 * This class can be use to represent any Eigen matrix within SOFA. It
 * implements (and overloads) the BaseMatrix class of Sofa. This class hence
 * allows to either to create a new Eigen matrix or copy/map an existing one
 * && use it in Sofa's components or visitors.
 *
 * \note The wrapper around Eigen sparse matrices has the following exceptions:
 *          1. Before the matrix is initialized using a series of triplets,
 *             accessing an element of the matrix has undefined behavior.
 *
 * Example:
 * \code{.cpp}
 *    // Wrapper by copy
 *    using EigenDense = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
 *    EigenDense m(100,100);
 *    m(30, 30) = 1;
 *
 *    EigenMatrix<EigenDense> wrapper(m);
 *    std::cout << m(30, 30) == wrapper(30, 30); // TRUE
 *
 *    wrapper(30, 30) = 100;
 *    std::cout << m(30, 30) == wrapper(30, 30); // FALSE
 * \endcode
 *
 * Example:
 * \code{.cpp}
 *    // Wrapper by reference
 *    using EigenDense = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
 *    EigenDense m(100,100);
 *    m(30, 30) = 1;
 *
 *    EigenMatrix<EigenDense &> wrapper(m);x
 *    std::cout << m(30, 30) == wrapper(30, 30); // TRUE
 *
 *    wrapper(30, 30) = 100;
 *    std::cout << m(30, 30) == wrapper(30, 30); // TRUE
 * \endcode
 */
template <typename Derived, typename Enable = void>
class EigenMatrix : public sofa::linearalgebra::BaseMatrix
{
    static_assert(
        std::is_base_of_v<Eigen::EigenBase<std::decay_t<Derived> >, std::decay_t<Derived> >,
        "The class template argument 'Derived' must be derived from Eigen::EigenBase<Derived>."
    );

public:
    using EigenType = std::remove_cv_t<std::remove_reference_t<Derived>>;
    using Base = sofa::linearalgebra::BaseMatrix;
    using Index = Base::Index;
    using Real = SReal;
    using Scalar = typename EigenType::Scalar;

    /**
     * Construct the class using another Eigen matrix. Depending on the template parameter used for the class,
     * this constructor will either create a new Eigen matrix && copy its content from the parameter eigen_matrix,
     * or it will simply store a reference to this external matrix.
     * @param eigen_matrix The external matrix
     */
    explicit EigenMatrix(std::remove_reference_t<Derived> & eigen_matrix) : p_eigen_matrix(eigen_matrix) {}

    /**
     * Construct a new rows x cols Eigen matrix of type Derived.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    EigenMatrix(Eigen::Index rows, Eigen::Index cols) : p_eigen_matrix(rows, cols) {}

    /**
     * Default constructor for an empty matrix.
     */
    EigenMatrix() = default;

    static const char* Name()
    {
        static std::string name = std::string("EigenMatrix") + sofa::type::trait::suffix_v<Scalar> ;
        return name.c_str();
    }

    // Abstract methods overrides
    inline Index rowSize() const final { return static_cast<Index>(p_eigen_matrix.rows()); }
    inline Index colSize() const final { return static_cast<Index>(p_eigen_matrix.cols()); }
    inline Real  element(Index i, Index j) const final { return static_cast<Real>(p_eigen_matrix(i,j)); }

    /** Resize the matrix to nbRow x nbCol dimensions. This method resets to zero all entries. */
    inline void  resize(Index nbRow, Index nbCol) final { p_eigen_matrix.setConstant(nbRow, nbCol, 0); }

    /** Set all entries to zero. Keeps the current matrix dimensions. */
    inline void  clear() override {p_eigen_matrix.setZero();}

    /** Set this value of the matrix entry (i, j) to the value of v. */
    inline void  set(Index i, Index j, double v) final {
        p_eigen_matrix(i, j) = static_cast<Scalar>(v);
    }

    /** Adds v to the value of the matrix entry (i, j). */
    inline void  add(Index i, Index j, double v) final {
        p_eigen_matrix(i, j) += static_cast<Scalar>(v);
    }

    // Block operations on 3x3 && 2x2 sub-matrices
    inline void add(Index i, Index j, const sofa::type::Mat3x3d & m) override { add_block<double, 3, 3>(i, j, m);}
    inline void add(Index i, Index j, const sofa::type::Mat3x3f & m) override { add_block<float, 3, 3>(i, j, m);}
    inline void add(Index i, Index j, const sofa::type::Mat2x2d & m) override { add_block<double, 2, 2>(i, j, m);}
    inline void add(Index i, Index j, const sofa::type::Mat2x2f & m) override { add_block<float, 2, 2>(i, j, m);}

    /** Sets the entire row i to zero */
    inline void clearRow(Index i) final {
        p_eigen_matrix.row(i).setZero();
    }

    /** Sets the rows from index imin to index imax (inclusively) to zero */
    inline void clearRows(Index imin, Index imax) final {
        p_eigen_matrix.middleRows(imin, (imax-imin)+1).setZero();
    }

    /** Sets the entire columns i to zero */
    inline void clearCol(Index i) final {
        p_eigen_matrix.col(i).setZero();
    }

    /** Sets the columns from index imin to index imax (inclusively) to zero */
    inline void clearCols(Index imin, Index imax) final {
        p_eigen_matrix.middleCols(imin, (imax-imin)+1).setZero();
    }

    /** Get a const reference to the underlying Eigen matrix  */
    const EigenType & matrix() const {return p_eigen_matrix;}

private:

    template <typename Scalar, unsigned int nrows, unsigned int ncols>
    void add_block(Index i, Index j, const sofa::type::Mat<nrows, ncols, Scalar> & m) {
        const Eigen::Map<const Eigen::Matrix<Scalar, nrows, ncols, Eigen::RowMajor>> block(&(m[0][0]));
        p_eigen_matrix.block(i, j, nrows, ncols) += block. template cast<typename EigenType::Scalar>();
    }

    ///< The actual Eigen Matrix.
    Derived p_eigen_matrix;
};

///////////////////////////////////
/// SparseMatrix specialization ///
///////////////////////////////////
template <typename Derived>
class EigenMatrix<Derived, std::enable_if_t<std::is_base_of_v<Eigen::SparseMatrixBase<std::decay_t<Derived>>, std::decay_t<Derived>>>> : public sofa::linearalgebra::BaseMatrix
{

public:
    using EigenType = std::remove_cv_t<std::remove_reference_t<Derived>>;
    using Base = sofa::linearalgebra::BaseMatrix;
    using Index = Base::Index;
    using Real = SReal;
    using Scalar = typename EigenType::Scalar;

    /**
     * Construct the class using another Eigen matrix. Depending on the template parameter used for the class,
     * this constructor will either create a new Eigen matrix && copy its content from the parameter eigen_matrix,
     * or it will simply store a reference to this external matrix.
     * @param eigen_matrix The external matrix
     */
    explicit EigenMatrix(std::remove_reference_t<Derived> & eigen_matrix) : p_eigen_matrix(eigen_matrix) {}

    /**
     * Construct a new rows x cols Eigen matrix of type Derived.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    EigenMatrix(Eigen::Index rows, Eigen::Index cols) : p_eigen_matrix(rows, cols) {}

    /**
     * Default constructor for an empty matrix.
     */
    EigenMatrix() = default;

    static const char* Name()
    {
        static std::string name = std::string("EigenSparseMatrix") + sofa::type::trait::suffix_v<Scalar> ;
        return name.c_str();
    }

    // Abstract methods overrides
    inline Index rowSize() const final { return static_cast<Index>(p_eigen_matrix.rows()); }
    inline Index colSize() const final { return static_cast<Index>(p_eigen_matrix.cols()); }

    /** States if the matrix is symmetric. Note that this value isn't set automatically, the user must
     * explicitly specify it using set_symmetric(true). When it is true, some optimizations will be enabled.
     */
    inline bool symmetric() const {return p_is_symmetric;}

    /**
     * Explicitly states if this matrix is symmetric.
     */
    inline void set_symmetric(bool is_symmetric) { p_is_symmetric = is_symmetric; }

    /**
     * @brief Return the matrix entry (i,j).
     * \warning If the matrix hasn't been initialized by calling compress() or set(), this
     *          method will always return 0.
     */
    inline Real  element(Index i, Index j) const final {
        assert(p_initialized && "Accessing an element on an uninitialized matrix.");
        return this->p_eigen_matrix.coeff(i,j);
    }

    /** Resize the matrix to nbRow x nbCol dimensions. This method resets to zero all entries. */
    inline void  resize(Index nbRow, Index nbCol) final {
        p_triplets.clear();
        this->p_eigen_matrix.resize(nbRow, nbCol);
        p_initialized = false;
    }

    /** Set all entries to zero. Keeps the current matrix dimensions. */
    inline void  clear() final {
        p_triplets.clear();
        this->p_eigen_matrix.setZero();
        p_initialized = false;
    }

    /**
     * Set this value of the matrix entry (i, j) to the value of v.
     *
     * \warning When the matrix hasn't been initialized yet, this method will have to do it. This
     *          means that further calls to set or add will be much slower && generate an uncompressed
     *          matrix. If you have to use the set method, make sure that all calls to add has been
     *          made beforehand.
     */
    inline void  set(Index i, Index j, double v) final {
        if (!p_initialized) {
            initialize();
        }

        this->p_eigen_matrix.coeffRef(i, j) = static_cast<Scalar>(v);
    }

    /**
     * Adds v to the value of the matrix entry (i, j).
     *
     * \warning If this method is called after the matrix has been initialized, the matrix will
     *          become uncompressed.
     */
    inline void  add(Index i, Index j, double v) final {
        // Note: this "if" condition should't slow down that much the addition of multiple entries
        //       because of branch predictions (the conditional branch will be same for the next
        //       X calls to add until compress is called).
        if (!p_initialized) {
            p_triplets.emplace_back(i, j, static_cast<Scalar>(v));
        } else {
            p_eigen_matrix.coeffRef(i, j) += static_cast<Scalar>(v);
        }
    }

    /**
     * Compress the matrix.
     *
     * If it is the first time that this method is called, then the matrix is built from a list of
     * triplets (i, j, value) accumulated during the calls of the methodes 'add' && 'set'.
     */
    void compress() final {
        if (!p_initialized) {
            initialize();
        } else {
            p_eigen_matrix.makeCompressed();
        }
    }

    // Block operations on 3x3 && 2x2 sub-matrices
    inline void add(Index i, Index j, const sofa::type::Mat3x3d & m) override { add_block<double, 3, 3>(i, j, m);}
    inline void add(Index i, Index j, const sofa::type::Mat3x3f & m) override { add_block<float, 3, 3>(i, j, m);}
    inline void add(Index i, Index j, const sofa::type::Mat2x2d & m) override { add_block<double, 2, 2>(i, j, m);}
    inline void add(Index i, Index j, const sofa::type::Mat2x2f & m) override { add_block<float, 2, 2>(i, j, m);}

    template<class EigenVectorDerived>
    EigenVector<EigenVectorDerived> operator*(const EigenVector<EigenVectorDerived>& v)
    {
        using EigenType = typename EigenVector<EigenVectorDerived>::EigenType;
        return EigenVector<EigenVectorDerived>(
            EigenType(p_eigen_matrix * v.vector()).eval());
    }

    /** Sets the entire row i to zero */
    inline void clearRow(Index row_id) final {
        if (!p_initialized) {
            initialize();
        }

        using StorageIndex = typename EigenType::StorageIndex;

        if constexpr (EigenType::IsRowMajor) {
            const auto & start = p_eigen_matrix.outerIndexPtr()[row_id];
            const auto & end   = p_eigen_matrix.outerIndexPtr()[row_id+1];
            std::memset(&(p_eigen_matrix.data().valuePtr()[start]), static_cast<int>(0), (end-start)*sizeof(Scalar));
        } else {
            for (int col_id=0; col_id<p_eigen_matrix.outerSize(); ++col_id) {
                const auto & start = p_eigen_matrix.outerIndexPtr()[col_id];
                const auto & end   = p_eigen_matrix.outerIndexPtr()[col_id+1];
                const auto id = p_eigen_matrix.data().searchLowerIndex(start, end, static_cast<StorageIndex>(row_id));
                if ((id<end) && (p_eigen_matrix.data().index(id)==row_id)) {
                    p_eigen_matrix.data().value(id) =  static_cast<const Scalar &>(0);
                }
            }
        }
    }

    /** Sets the rows from index imin to index imax (inclusively) to zero */
    inline void clearRows(Index imin, Index imax) final {
        if (!p_initialized) {
            initialize();
        }

        using StorageIndex = typename EigenType::StorageIndex;

        if constexpr (EigenType::IsRowMajor) {
            const auto & start = p_eigen_matrix.outerIndexPtr()[imin];
            const auto & end   = p_eigen_matrix.outerIndexPtr()[imax+1];
            std::memset(&(p_eigen_matrix.data().valuePtr()[start]), static_cast<int>(0), (end-start)*sizeof(Scalar));
        } else {
            for (int col_id=0; col_id<p_eigen_matrix.outerSize(); ++col_id) {
                const auto & start = p_eigen_matrix.outerIndexPtr()[col_id];
                const auto & end   = p_eigen_matrix.outerIndexPtr()[col_id+1];
                auto id = p_eigen_matrix.data().searchLowerIndex(start, end, static_cast<StorageIndex>(imin));
                for (;id<end;++id) {
                    if (imin <= p_eigen_matrix.data().index(id) && p_eigen_matrix.data().index(id) <= imax) {
                        p_eigen_matrix.data().value(id) =  static_cast<const Scalar &>(0);
                    } else if (p_eigen_matrix.data().index(id) > imax) {
                        break;
                    }
                }
            }
        }
    }

    /** Sets the entire columns i to zero */
    inline void clearCol(Index col_id) final {
        if (!p_initialized) {
            initialize();
        }

        using StorageIndex = typename EigenType::StorageIndex;

        if constexpr (!EigenType::IsRowMajor) {
            const auto & start = p_eigen_matrix.outerIndexPtr()[col_id];
            const auto & end   = p_eigen_matrix.outerIndexPtr()[col_id+1];
            std::memset(&(p_eigen_matrix.data().valuePtr()[start]), static_cast<int>(0), (end-start)*sizeof(Scalar));
        } else {
            for (int row_id=0; row_id<p_eigen_matrix.outerSize(); ++row_id) {
                const auto & start = p_eigen_matrix.outerIndexPtr()[row_id];
                const auto & end   = p_eigen_matrix.outerIndexPtr()[row_id+1];
                const auto id = p_eigen_matrix.data().searchLowerIndex(start, end, static_cast<StorageIndex>(col_id));
                if ((id<end) && (p_eigen_matrix.data().index(id)==col_id)) {
                    p_eigen_matrix.data().value(id) =  static_cast<const Scalar &>(0);
                }
            }
        }
    }

    /** Sets the columns from index imin to index imax (inclusively) to zero */
    inline void clearCols(Index imin, Index imax) final {
        if (!p_initialized) {
            initialize();
        }

        using StorageIndex = typename EigenType::StorageIndex;

        if constexpr (!EigenType::IsRowMajor) {
            const auto & start = p_eigen_matrix.outerIndexPtr()[imin];
            const auto & end   = p_eigen_matrix.outerIndexPtr()[imax+1];
            std::memset(&(p_eigen_matrix.data().valuePtr()[start]), static_cast<int>(0), (end-start)*sizeof(Scalar));
        } else {
            for (int row_id=0; row_id<p_eigen_matrix.outerSize(); ++row_id) {
                const auto & start = p_eigen_matrix.outerIndexPtr()[row_id];
                const auto & end   = p_eigen_matrix.outerIndexPtr()[row_id+1];
                auto id = p_eigen_matrix.data().searchLowerIndex(start, end, static_cast<StorageIndex>(imin));
                for (;id<end;++id) {
                    if (imin <= p_eigen_matrix.data().index(id) && p_eigen_matrix.data().index(id) <= imax) {
                        p_eigen_matrix.data().value(id) =  static_cast<const Scalar &>(0);
                    } else if (p_eigen_matrix.data().index(id) > imax) {
                        break;
                    }
                }
            }
        }
    }

    /** Sets the entire row i && column i to zero */
    inline void clearRowCol(Index i) final {
        if (!p_initialized) {
            initialize();
        }

        using StorageIndex = typename EigenType::StorageIndex;

        // Clear the complete inner vector i (the row i or column i if it is in row major or column major respectively)
        const auto & start = p_eigen_matrix.outerIndexPtr()[i];
        const auto & end   = p_eigen_matrix.outerIndexPtr()[i+1];
        std::memset(&(p_eigen_matrix.data().valuePtr()[start]), static_cast<int>(0), (end-start)*sizeof(Scalar));

        // Next, to clear the ith inner coefficients of each row in row-major (each column in column-major)
        if (symmetric() && p_eigen_matrix.rows() == p_eigen_matrix.cols()) {
            // When the sparse matrix is symmetric, we can avoid doing the binary search on each inner vectors. Instead,
            // We loop on each inner coefficient of the ith outer vector (eg each (i,j) of the ith row in row major)
            // && do the binary search only on the jth outer vector
            for (typename EigenType::InnerIterator it(p_eigen_matrix, i); it; ++it) {
                const auto inner_index = EigenType::IsRowMajor ? it.col() : it.row();
                const auto & start = p_eigen_matrix.outerIndexPtr()[inner_index];
                const auto & end   = p_eigen_matrix.outerIndexPtr()[inner_index+1];
                const auto id = p_eigen_matrix.data().searchLowerIndex(start, end, static_cast<StorageIndex>(i));
                if ((id<end) && (p_eigen_matrix.data().index(id)==i)) {
                    p_eigen_matrix.data().value(id) =  static_cast<const Scalar &>(0);
                }
            }
        } else {
            for (int outer_id=0; outer_id<p_eigen_matrix.outerSize(); ++outer_id) {
                const auto & start = p_eigen_matrix.outerIndexPtr()[outer_id];
                const auto & end   = p_eigen_matrix.outerIndexPtr()[outer_id+1];
                const auto id = p_eigen_matrix.data().searchLowerIndex(start, end, static_cast<StorageIndex>(i));
                if ((id<end) && (p_eigen_matrix.data().index(id)==i)) {
                    p_eigen_matrix.data().value(id) =  static_cast<const Scalar &>(0);
                }
            }
        }
    }

    /** Get a const reference to the underlying Eigen matrix  */
    const Derived & matrix() const {return p_eigen_matrix;}
private:

    /**
     * Block addition with a NxC matrix
     */
    template <typename Scalar, unsigned int N, unsigned int C>
    void add_block(Index i, Index j, const sofa::type::Mat<N, C, Scalar> & m) {
        using StorageIndex = typename Eigen::SparseMatrix<typename EigenType::Scalar>::StorageIndex;
        for (unsigned int k=0;k<N;++k) {
            for (unsigned int l=0;l<C;++l) {
                const auto value = static_cast<typename EigenType::Scalar>(m[k][l]);
                if (!p_initialized) {
                    p_triplets.emplace_back(static_cast<StorageIndex>(i+k), static_cast<StorageIndex>(j+l), value);
                } else {
                    p_eigen_matrix.coeffRef(i+k, j+l) += value;
                }
            }
        }
    }

    /**
     * @brief initialize the matrix with the accumulated triplets. The resulting matrix will be
     *        compressed. Once this is done, the operation add && addblock will be much slower
     *        if the entries to be added were zero before hand.
     */
    void initialize() {
        p_eigen_matrix.setFromTriplets(p_triplets.begin(), p_triplets.end());
        p_triplets.clear();
        p_initialized = true;
    }

    ///< Triplets are used to store matrix entries before the call to 'compress'.
    /// Duplicates entries are summed up.
    std::vector<Eigen::Triplet<typename EigenType::Scalar>> p_triplets;

    ///< Whether or not the matrix has been initialized with triplets yet. This will determined the behavior
    /// of the add && set methods. When the matrix hasn't been initialized, the add && set methods will simply append
    /// the new coefficient to the triplet list.
    bool p_initialized = false;

    ///< The actual Eigen Matrix.
    Derived p_eigen_matrix;

    ///< States if the matrix is symmetric. Note that this value isn't set automatically, the user must
    ///< explicitly specify it using set_symmetric(true). When it is true, some optimizations will be enabled.
    bool p_is_symmetric = false;
};

#if !defined(SOFABASELINEARSOLVER_EIGEN_EIGENMATRIX_CPP)
    extern template class SOFA_LINEARALGEBRA_API EigenMatrix<Eigen::SparseMatrix<SReal, Eigen::RowMajor> >;
#endif

} // namespace sofa::linearalgebra::eigen
