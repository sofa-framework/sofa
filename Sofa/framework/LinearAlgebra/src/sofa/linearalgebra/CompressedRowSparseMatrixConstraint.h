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
#pragma once

#include <sofa/linearalgebra/config.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixGeneric.h>

#include <numeric>

namespace sofa::linearalgebra
{

template<class RowType, class VecDeriv, typename Real = typename VecDeriv::value_type::Real>
Real CompressedRowSparseMatrixVecDerivMult(const RowType row, const VecDeriv& vec)
{
    Real r = 0;
    for (typename RowType::first_type it = row.first, itend = row.second; it != itend; ++it)
        r += it.val() * vec[it.index()];
    return r;
}

template<class RowType, class VecDeriv>
void convertCompressedRowSparseMatrixRowToVecDeriv(const RowType row, VecDeriv& out)
{
    for (typename RowType::first_type it = row.first, itend = row.second; it != itend; ++it)
    {
        out[it.index()] += it.val();
    }
}

/// Constraint policy type, showing the types and flags to give to CompressedRowSparseMatrix
/// for its second template type. The default values correspond to the original implementation.
class CRSConstraintPolicy : public linearalgebra::CRSDefaultPolicy
{
public:
    static constexpr bool AutoSize = true;
    static constexpr bool AutoCompress = true;
    static constexpr bool CompressZeros = false;
    static constexpr bool ClearByZeros = false;
    static constexpr bool OrderedInsertion = false;

    static constexpr int  matrixType = 2;
};

template<typename TBlock, typename TPolicy = CRSConstraintPolicy >
class CompressedRowSparseMatrixConstraint : public sofa::linearalgebra::CompressedRowSparseMatrixGeneric<TBlock, TPolicy>
{
public:
    typedef CompressedRowSparseMatrixConstraint<TBlock, TPolicy> Matrix;
    typedef linearalgebra::CompressedRowSparseMatrixGeneric<TBlock, TPolicy> CRSMatrix;
    typedef typename CRSMatrix::Policy Policy;

    using Block     = TBlock;
    using VecBlock  = typename linearalgebra::CRSBlockTraits<Block>::VecBlock;
    using VecIndex = typename linearalgebra::CRSBlockTraits<Block>::VecIndex;
    using VecFlag  = typename linearalgebra::CRSBlockTraits<Block>::VecFlag;
    using Index    = typename VecIndex::value_type;
    static constexpr Index s_invalidIndex = std::is_signed_v<Index> ? std::numeric_limits<Index>::lowest() : std::numeric_limits<Index>::max();

    typedef typename CRSMatrix::Block Data;
    typedef typename CRSMatrix::Range Range;
    typedef typename CRSMatrix::traits traits;
    typedef typename CRSMatrix::Real Real;
    typedef typename CRSMatrix::Index KeyType;
    typedef typename CRSMatrix::IndexedBlock IndexedBlock;

public:
    CompressedRowSparseMatrixConstraint()
        : CRSMatrix()
    {
    }

    CompressedRowSparseMatrixConstraint(Index nbRow, Index nbCol)
        : CRSMatrix(nbRow, nbCol)
    {
    }

    bool empty() const
    {
        return this->rowIndex.empty();
    }

    class RowType;
    class RowConstIterator;
    /// Row Sparse Matrix columns constant Iterator to match with constraint matrix manipulation
    class ColConstIterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = Index;
        using difference_type = Index;
        using pointer = Index*;
        using reference = Index&;

        friend class RowConstIterator;
        friend class RowType;
    protected:

        ColConstIterator(const Index _rowIt, int _internal, const CompressedRowSparseMatrixConstraint* _matrix)
            : m_rowIt(_rowIt)
            , m_internal(_internal)
            , m_matrix(_matrix)
        {}

    public:

        ColConstIterator(const ColConstIterator& it2)
            : m_rowIt(it2.m_rowIt)
            , m_internal(it2.m_internal)
            , m_matrix(it2.m_matrix)
        {}

        ColConstIterator& operator=(const ColConstIterator& other)
        {
            if (this != &other)
            {
                m_rowIt = other.m_rowIt;
                m_internal = other.m_internal;
                m_matrix = other.m_matrix;
            }
            return *this;
        }

        Index row() const
        {
            return m_matrix->rowIndex[m_rowIt];
        }

        /// @return the constraint value
        const TBlock &val() const
        {
            return m_matrix->colsValue[m_internal];
        }

        /// @return the DOF index the constraint is applied on
        Index index() const
        {
            return m_matrix->colsIndex[m_internal];
        }

        const Index getInternal() const
        {
            return m_internal;
        }

        bool isInvalid() const
        {
            return m_internal == CompressedRowSparseMatrixConstraint::s_invalidIndex;
        }

        void operator++() // prefix
        {
            m_internal++;
        }

        void operator++(int) // postfix
        {
            m_internal++;
        }

        void operator--() // prefix
        {
            m_internal--;
        }

        void operator--(int) // postfix
        {
            m_internal--;
        }

        void operator+=(int i)
        {
            m_internal += i;
        }

        void operator-=(int i)
        {
            m_internal -= i;
        }

        bool operator==(const ColConstIterator& it2) const
        {
            return (m_internal == it2.m_internal);
        }

        bool operator!=(const ColConstIterator& it2) const
        {
            return (m_internal != it2.m_internal);
        }

        bool operator<(const ColConstIterator& it2) const
        {
            return m_internal < it2.m_internal;
        }

        bool operator>(const ColConstIterator& it2) const
        {
            return m_internal > it2.m_internal;
        }

    private :

        Index m_rowIt;
        Index m_internal;
        const CompressedRowSparseMatrixConstraint* m_matrix;
    };

    class RowConstIterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = Index;
        using difference_type = Index;
        using pointer = Index*;
        using reference = Index&;

        friend class CompressedRowSparseMatrixConstraint;

    protected:

        RowConstIterator(const CompressedRowSparseMatrixConstraint* _matrix, int _m_internal)
            : m_internal(_m_internal)
            , m_matrix(_matrix)
        {}

    public:

        RowConstIterator(const RowConstIterator& it2)
            : m_internal(it2.m_internal)
            , m_matrix(it2.m_matrix)
        {}

        RowConstIterator()
        {}

        RowConstIterator&  operator=(const RowConstIterator& other)
        {
            if (this != &other)
            {
                m_matrix = other.m_matrix;
                m_internal = other.m_internal;
            }
            return *this;
        }

        Index index() const
        {
            return m_matrix->rowIndex[m_internal];
        }

        Index getInternal() const
        {
            return m_internal;
        }

        bool isInvalid() const
        {
            return m_internal == CompressedRowSparseMatrixConstraint::s_invalidIndex;
        }

        ColConstIterator begin() const
        {
            if (isInvalid())
            {
                return ColConstIterator(m_internal, s_invalidIndex, m_matrix);
            }
            Range r = m_matrix->getRowRange(m_internal);
            return ColConstIterator(m_internal, r.begin(), m_matrix);
        }

        ColConstIterator end() const
        {
            if (isInvalid())
            {
                return ColConstIterator(m_internal, s_invalidIndex, m_matrix);
            }
            Range r = m_matrix->getRowRange(m_internal);
            return ColConstIterator(m_internal, r.end(), m_matrix);
        }

        RowType row() const
        {
            Range r = m_matrix->getRowRange(m_internal);
            return RowType(ColConstIterator(m_internal, r.begin(), m_matrix),
                           ColConstIterator(m_internal, r.end(), m_matrix));
        }

        bool empty() const
        {
            Range r = m_matrix->getRowRange(m_internal);
            return r.empty();
        }

        void operator++() // prefix
        {
            m_internal++;
        }

        void operator++(int) // postfix
        {
            m_internal++;
        }

        void operator--() // prefix
        {
            m_internal--;
        }

        void operator--(int) // postfix
        {
            m_internal--;
        }

        void operator+=(int i)
        {
            m_internal += i;
        }

        void operator-=(int i)
        {
            m_internal -= i;
        }

        int operator-(const RowConstIterator& it2) const
        {
            return m_internal - it2.m_internal;
        }

        RowConstIterator operator+(int i) const
        {
            RowConstIterator res = *this;
            res += i;
            return res;
        }

        RowConstIterator operator-(int i) const
        {
            RowConstIterator res = *this;
            res -= i;
            return res;
        }

        bool operator==(const RowConstIterator& it2) const
        {
            return m_internal == it2.m_internal;
        }

        bool operator!=(const RowConstIterator& it2) const
        {
            return !(m_internal == it2.m_internal);
        }

        bool operator<(const RowConstIterator& it2) const
        {
            return m_internal < it2.m_internal;
        }

        bool operator>(const RowConstIterator& it2) const
        {
            return m_internal > it2.m_internal;
        }

        template <class VecDeriv, typename Real>
        Real operator*(const VecDeriv& v) const
        {
            return CompressedRowSparseMatrixVecDerivMult(row(), v);
        }

    private:

        Index m_internal;
        const CompressedRowSparseMatrixConstraint* m_matrix;
    };

    /// Get the iterator corresponding to the beginning of the rows of blocks
    RowConstIterator begin() const
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method !
        return RowConstIterator(this,
            this->rowIndex.empty() ? s_invalidIndex : 0);
    }

    /// Get the iterator corresponding to the end of the rows of blocks
    RowConstIterator end() const
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method !
        return RowConstIterator(this,
            this->rowIndex.empty() ? s_invalidIndex : Index(this->rowIndex.size()));
    }

    /// Get the iterator corresponding to the beginning of the rows of blocks
    RowConstIterator cbegin() const
    {
        if constexpr(Policy::AutoCompress) const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method !
        return RowConstIterator(this, 0);
    }

    /// Get the iterator corresponding to the end of the rows of blocks
    RowConstIterator cend() const
    {
        if constexpr(Policy::AutoCompress) const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method !
        return RowConstIterator(this, Index(this->rowIndex.size()));
    }

    class RowWriteAccessor
    {
    public:

        friend class CompressedRowSparseMatrixConstraint;

    protected:

        RowWriteAccessor(CompressedRowSparseMatrixConstraint* _matrix, int _rowIndex)
            : m_rowIndex(_rowIndex)
            , m_matrix(_matrix)
        {}

    public:

        void addCol(Index id, const Block& value)
        {
            *m_matrix->wblock(m_rowIndex, id, true) += value;
        }

        // TODO: this is wrong in case the returned block is within the uncompressed triplets
        void setCol(Index id, const Block& value)
        {
            *m_matrix->wblock(m_rowIndex, id, true) = value;
        }

        bool operator==(const RowWriteAccessor& it2) const
        {
            return m_rowIndex == it2.m_rowIndex;
        }

        bool operator!=(const RowWriteAccessor& it2) const
        {
            return !(m_rowIndex == it2.m_rowIndex);
        }

    private:
        int m_rowIndex;
        CompressedRowSparseMatrixConstraint* m_matrix;
    };

    class RowType : public std::pair<ColConstIterator, ColConstIterator>
    {
        typedef std::pair<ColConstIterator, ColConstIterator> Inherit;
    public:
        RowType(ColConstIterator begin, ColConstIterator end) : Inherit(begin,end) {}
        ColConstIterator begin() const { return this->first; }
        ColConstIterator end() const { return this->second; }
        ColConstIterator cbegin() const { return this->first; }
        ColConstIterator cend() const { return this->second; }
        void setBegin(ColConstIterator i) { this->first = i; }
        void setEnd(ColConstIterator i) { this->second = i; }
        bool empty() const { return begin() == end(); }
        Index size() const { return end().getInternal() - begin().getInternal(); }
        void operator++() { ++this->first; }
        void operator++(int) { ++this->first; }
        ColConstIterator find(Index col) const
        {
            const CompressedRowSparseMatrixConstraint* matrix = this->first.m_matrix;
            Range r(this->first.m_internal, this->second.m_internal);
            Index index = 0;
            if (!matrix->sortedFind(matrix->colsIndex, r, col, index))
            {
                index = r.end(); // not found -> return end
            }
            return ColConstIterator(this->first.m_rowIt, index, matrix);
        }

    };

    /// Get the number of constraint
    size_t size() const
    {
        if constexpr(Policy::AutoCompress) const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method !
        return this->getRowIndex().size();
    }

    /// @return Constant Iterator on specified row
    /// @param lIndex row index
    /// If lIndex row doesn't exist, returns end iterator
    RowConstIterator readLine(Index lIndex) const
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method !
        Index rowId = (this->nBlockRow == 0) ? 0 : lIndex * this->rowIndex.size() / this->nBlockRow;
        if (this->sortedFind(this->rowIndex, lIndex, rowId))
        {
            return RowConstIterator(this, rowId);
        }
        else
        {
            return RowConstIterator(this, this->rowIndex.size());
        }
    }

    /// @return Iterator on specified row
    /// @param lIndex row index
    RowWriteAccessor writeLine(Index lIndex)
    {
        return RowWriteAccessor(this, lIndex);
    }

    /// @param lIndex row Index
    /// @param row constraint itself
    /// If lindex already exists, overwrite existing constraint
    void setLine(Index lIndex, RowType row)
    {
        if (readLine(lIndex) != this->end()) this->clearRowBlock(lIndex);

        RowWriteAccessor it(this, lIndex);
        ColConstIterator colIt = row.first;
        ColConstIterator colItEnd = row.second;

        while (colIt != colItEnd)
        {
            it.setCol(colIt.index(), colIt.val());
            ++colIt;
        }
    }

    /// @param lIndex row Index
    /// @param row constraint itself
    /// If lindex doesn't exists, creates the row
    void addLine(Index lIndex, RowType row)
    {
        RowWriteAccessor it(this, lIndex);

        ColConstIterator colIt = row.first;
        ColConstIterator colItEnd = row.second;

        while (colIt != colItEnd)
        {
            it.addCol(colIt.index(), colIt.val());
            ++colIt;
        }
    }

    template< class VecDeriv>
    void multTransposeBaseVector(VecDeriv& res, const sofa::linearalgebra::BaseVector* lambda ) const
    {
        typedef typename VecDeriv::value_type Deriv;

        static_assert(std::is_same<Deriv, TBlock>::value, "res must be contain same type as CompressedRowSparseMatrix type");

        for (auto rowIt = begin(), rowItEnd = end(); rowIt != rowItEnd; ++rowIt)
        {
            const SReal f = lambda->element(rowIt.index());
            for (auto colIt = rowIt.begin(), colItEnd = rowIt.end(); colIt != colItEnd; ++colIt)
            {
                res[colIt.index()] += colIt.val() * f;
            }
        }
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const CompressedRowSparseMatrixConstraint<TBlock, Policy>& sc)
    {
        for (RowConstIterator rowIt = sc.begin(); rowIt !=  sc.end(); ++rowIt)
        {
            out << "Constraint ID : ";
            out << rowIt.index();
            for (ColConstIterator colIt = rowIt.begin(); colIt !=  rowIt.end(); ++colIt)
            {
                out << "  dof ID : " << colIt.index() << "  value : " << colIt.val() << "  ";
            }
            out << "\n";
        }

        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, CompressedRowSparseMatrixConstraint<TBlock, Policy>& sc)
    {
        sc.clear();

        unsigned int c_id;
        unsigned int c_number;
        unsigned int c_dofIndex;
        TBlock c_value;

        while (!(in.rdstate() & std::istream::eofbit))
        {
            in >> c_id;
            in >> c_number;

            auto c_it = sc.writeLine(c_id);

            for (unsigned int i = 0; i < c_number; i++)
            {
                in >> c_dofIndex;
                in >> c_value;
                c_it.addCol(c_dofIndex, c_value);
            }
        }

        sc.compress();
        return in;
    }

    static const char* Name()
    {
        static std::string name = std::string("CompressedRowSparseMatrixConstraint") + std::string(traits::Name());
        return name.c_str();
    }

    /// Definition for MapMapSparseMatrix and CompressedRowSparseMatrixConstraint compatibility
    using ColIterator = ColConstIterator;
    using RowIterator = RowWriteAccessor;
};

#if !defined(SOFA_LINEARALGEBRA_COMPRESSEDROWSPARSEMATRIXCONSTRAINT_CPP) 

extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec1f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec2f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec3f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec6f>;


extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec1d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec2d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec3d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec6d>;

#endif

} // namespace sofa::linearalgebra
