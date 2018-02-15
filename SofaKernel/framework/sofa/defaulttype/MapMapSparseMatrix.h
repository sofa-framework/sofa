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
#ifndef SOFA_DEFAULTTYPE_MAPMAPSPARSEMATRIX_H
#define SOFA_DEFAULTTYPE_MAPMAPSPARSEMATRIX_H

#include <map>
#include "BaseVector.h"

namespace sofa
{

namespace defaulttype
{

template<class MatrixRow, class VecDeriv>
typename VecDeriv::Real SparseMatrixVecDerivMult(const MatrixRow& row, const VecDeriv& vec)
{
    typename VecDeriv::Real r = 0;
    for (typename MatrixRow::const_iterator it = row.begin(), itend = row.end(); it != itend; ++it)
        r += it->second * vec[it->first];
    return r;
}

template <class T>
class MapMapSparseMatrix
{
public:
    typedef T Data;
    typedef unsigned int KeyType;
    typedef typename std::map< KeyType, T > RowType;

    /// Removes every matrix elements
    void clear()
    {
        m_data.clear();
    }

    /// @return true if the matrix is empty
    bool empty() const
    {
        return m_data.empty();
    }

    /// @return the number of rows
    std::size_t size() const
    {
        return m_data.size();
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const MapMapSparseMatrix<T>& sc)
    {
        for (typename SparseMatrix::const_iterator rowIt = sc.m_data.begin(); rowIt !=  sc.m_data.end(); ++rowIt)
        {
            out << rowIt->first;
            out << " ";
            out << rowIt->second.size();
            out << " ";
            for (typename RowType::const_iterator colIt = rowIt->second.begin(); colIt !=  rowIt->second.end(); ++colIt)
            {
                out << colIt->first << " " << colIt->second << "  ";
            }
            out << "\n";
        }

        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, MapMapSparseMatrix<T>& sc)
    {
        sc.clear();

        unsigned int c_id;
        unsigned int c_number;
        unsigned int c_dofIndex;
        T c_value;

        while (!(in.rdstate() & std::istream::eofbit))
        {
            in >> c_id;
            in >> c_number;

            RowIterator c_it = sc.writeLine(c_id);

            for (unsigned int i = 0; i < c_number; i++)
            {
                in >> c_dofIndex;
                in >> c_value;
                c_it.addCol(c_dofIndex, c_value);
            }
        }

        return in;
    }

    template< class VecDeriv>
    void multTransposeBaseVector(VecDeriv& res, const sofa::defaulttype::BaseVector* lambda ) const
    {
        typedef typename VecDeriv::value_type Deriv;

        static_assert(std::is_same<Deriv, T>::value, "res must contain same type as MapMapSparseMatrix type");

        for (auto rowIt = begin(), rowItEnd = end(); rowIt != rowItEnd; ++rowIt)
        {
            const SReal f = lambda->element(rowIt.index());
            for (auto colIt = rowIt.begin(), colItEnd = rowIt.end(); colIt != colItEnd; ++colIt)
            {
                res[colIt.index()] += colIt.val() * f;
            }
        }
    }

protected:

    typedef std::map< KeyType, RowType > SparseMatrix;

    /// Data container
    SparseMatrix m_data;

public:
    class RowConstIterator;

    /// Sparse Matrix columns constant Iterator
    class ColConstIterator
    {
    public:

        typedef typename RowType::key_type KeyT;
        typedef typename RowType::const_iterator Iterator;

        friend class RowConstIterator;

    protected:

        /*ColConstIterator()
        	{

        }*/

        ColConstIterator(Iterator _internal, const KeyT _rowIndex)
            : m_internal(_internal)
            , m_rowIndex(_rowIndex)
        {

        }

    public:

        ColConstIterator(const ColConstIterator& it2)
            : m_internal(it2.m_internal)
            , m_rowIndex(it2.m_rowIndex)
        {

        }

        void operator=(const ColConstIterator& it2)
        {
            m_internal = it2.m_internal;
            m_rowIndex = it2.m_rowIndex;
        }

        /// @return the row index of the parsed row (ie constraint id)
        typename SparseMatrix::key_type row() const
        {
            return m_rowIndex;
        }

        /// @return the DOF index the constraint is applied on
        KeyT index() const
        {
            return m_internal->first;
        }

        /// @return the constraint value
        const T &val() const
        {
            return m_internal->second;
        }

        /// @return the DOF index the constraint is applied on and its value
        const std::pair< KeyT, T >& operator*() const
        {
            return *m_internal;
        }

        /// @return the DOF index the constraint is applied on and its value
        const std::pair< KeyT, T >& operator->() const
        {
            return *m_internal;
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

        bool operator==(const ColConstIterator& it2) const
        {
            return m_internal == it2.m_internal;
        }

        bool operator!=(const ColConstIterator& it2) const
        {
            return !(m_internal == it2.m_internal);
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

        Iterator m_internal;
        const KeyT m_rowIndex;
    };


    class RowConstIterator
    {
    public:

        typedef typename SparseMatrix::const_iterator Iterator;
        typedef typename SparseMatrix::key_type KeyT;

        template <class U> friend class MapMapSparseMatrix;

    protected:

        /*RowConstIterator()
        	{

        }*/

        RowConstIterator(Iterator _internal)
            : m_internal(_internal)
        {

        }

    public:

        RowConstIterator(const RowConstIterator& it2)
            : m_internal(it2.m_internal)
        {

        }

        ColConstIterator begin()
        {
            return ColConstIterator(m_internal->second.begin(), m_internal->first);
        }

        ColConstIterator end()
        {
            return ColConstIterator(m_internal->second.end(), m_internal->first);
        }

        void operator=(const RowConstIterator& it2)
        {
            m_internal = it2.m_internal;
        }

        const std::pair< KeyT, RowType >& operator*() const
        {
            return *m_internal;
        }

        ///@
        const KeyT index() const
        {
            return m_internal->first;
        }

        const RowType& row() const
        {
            return m_internal->second;
        }


        const std::pair< KeyT, RowType >& operator->() const
        {
            return *m_internal;
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

        template <class VecDeriv>
        typename VecDeriv::Real operator*(const VecDeriv& v) const
        {
            return SparseMatrixVecDerivMult(row(), v);
        }

    private:

        Iterator m_internal;
    };


    RowConstIterator begin() const
    {
        return RowConstIterator(this->m_data.begin());
    }

    RowConstIterator end() const
    {
        return RowConstIterator(this->m_data.end());
    }

    class RowIterator;

    class ColIterator
    {
    public:

        typedef typename RowType::iterator Iterator;
        typedef typename RowType::key_type KeyT;

        friend class RowIterator;

    protected:

        /*ColIterator()
        	{

        }*/

        ColIterator(Iterator _internal, const KeyT _rowIndex)
            : m_internal(_internal)
            , m_rowIndex(_rowIndex)
        {

        }

    public:

        ColIterator(const ColIterator& it2)
            : m_internal(it2.m_internal)
            , m_rowIndex(it2.m_rowIndex)
        {

        }

        void operator=(const ColIterator& it2)
        {
            m_internal = it2.m_internal;
            m_rowIndex = it2.m_rowIndex;
        }

        /// @return the row index of the parsed row (ie constraint id)
        typename SparseMatrix::key_type row() const
        {
            return m_rowIndex;
        }

        /// @return the DOF index the constraint is applied on
        KeyT index() const
        {
            return m_internal->first;
        }

        /// @return the constraint value
        T &val()
        {
            return m_internal->second;
        }

        /// @return the DOF index the constraint is applied on and its value
        std::pair< KeyT, T >& operator*()
        {
            return *m_internal;
        }

        /// @return the DOF index the constraint is applied on and its value
        std::pair< KeyT, T >& operator->()
        {
            return *m_internal;
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

        bool operator==(const ColIterator& it2) const
        {
            return m_internal == it2.m_internal;
        }

        bool operator!=(const ColIterator& it2) const
        {
            return !(m_internal == it2.m_internal);
        }

        bool operator<(const ColIterator& it2) const
        {
            return m_internal < it2.m_internal;
        }

        bool operator>(const ColIterator& it2) const
        {
            return m_internal > it2.m_internal;
        }

    private :

        Iterator m_internal;
        const KeyT m_rowIndex;
    };


    class RowIterator
    {
    public:
        typedef typename SparseMatrix::key_type KeyT;
        typedef typename SparseMatrix::iterator Iterator;

        template <class U> friend class MapMapSparseMatrix;

    protected:

        /*RowIterator()
        	{

        }*/

        RowIterator(Iterator _internal)
            : m_internal(_internal)
        {

        }

    public:

        RowIterator(const RowIterator& it2)
            : m_internal(it2.m_internal)
        {

        }

        ColIterator begin()
        {
            return ColIterator(m_internal->second.begin(), m_internal->first);
        }

        ColIterator end()
        {
            return ColIterator(m_internal->second.end(), m_internal->first);
        }

        void operator=(const RowIterator& it2)
        {
            m_internal = it2.m_internal;
        }

        std::pair< KeyT, RowType >& operator*()
        {
            return *m_internal;
        }

        std::pair< KeyT, RowType >& operator->()
        {
            return *m_internal;
        }

        KeyT index()
        {
            return m_internal->first;
        }

        RowType& row()
        {
            return m_internal->second;
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

        bool operator==(const RowIterator& it2) const
        {
            return m_internal == it2.m_internal;
        }

        bool operator!=(const RowIterator& it2) const
        {
            return !(m_internal == it2.m_internal);
        }

        bool operator<(const RowIterator& it2) const
        {
            return m_internal < it2.m_internal;
        }

        bool operator>(const RowIterator& it2) const
        {
            return m_internal > it2.m_internal;
        }

        void addCol(KeyT id, const T& value)
        {
            RowType& row = m_internal->second;
            typename RowType::iterator it = row.find(id);

            if (it != row.end())
            {
                it->second += value;
            }
            else
            {
                row.insert(std::make_pair(id, value));
            }
        }

        void setCol(KeyT id, const T& value)
        {
            RowType& row = m_internal->second;
            typename RowType::iterator it = row.find(id);

            if (it != row.end())
            {
                it->second = value;
            }
            else
            {
                row.insert(std::make_pair(id, value));
            }
        }

    private:

        Iterator m_internal;
    };


    RowIterator begin()
    {
        return RowIterator(this->m_data.begin());
    }

    RowIterator end()
    {
        return RowIterator(this->m_data.end());
    }

    /// @return Constant Iterator on specified row
    /// @param lIndex row index
    /// If lIndex row doesn't exist, returns end iterator
    RowConstIterator readLine(KeyType lIndex) const
    {
        return RowConstIterator(m_data.find(lIndex));
    }

    /// @return Iterator on specified row
    /// @param lIndex row index
    /// If lIndex row doesn't exist, creates the line and returns an iterator on it
    RowIterator writeLine(KeyType lIndex)
    {
        RowIterator it(m_data.find(lIndex));

        if (it != this->end())
        {
            return it;
        }
        else
        {
            std::pair< typename SparseMatrix::iterator, bool > res = m_data.insert(std::make_pair(lIndex, RowType()));
            return RowIterator(res.first);
        }
    }

    /// @return Pair of Iterator on specified row and boolean on true if insertion took place
    /// @param lIndex row Index
    /// @param row constraint itself
    /// If lindex already exists, overwrite existing constraint
    std::pair< RowIterator, bool > writeLine(KeyType lIndex, RowType row)
    {
        RowIterator it(m_data.find(lIndex));

        if (it != this->end())
        {
            // removes already existing constraint
            m_data.erase(m_data.find(lIndex));
        }

        std::pair< typename SparseMatrix::iterator, bool > res = m_data.insert(std::make_pair(lIndex, row));
        return std::make_pair(RowIterator(res.first), res.second);
    }

    /// @return Pair of Iterator on specified row and boolean on true if addition took place
    /// @param lIndex row Index
    /// @param row constraint itself
    /// If lindex doesn't exists, creates the row
    std::pair< RowIterator, bool > addLine(KeyType lIndex, RowType row)
    {
        RowIterator it(m_data.find(lIndex));

        if (it == this->end())
        {
            std::pair< typename SparseMatrix::iterator, bool > res = m_data.insert(std::make_pair(lIndex, row));
            return std::make_pair(RowIterator(res.first), res.second);
        }
        else
        {
            typename RowType::const_iterator rowIt = row.begin();
            typename RowType::const_iterator rowItEnd = row.end();

            while (rowIt != rowItEnd)
            {
                it.addCol(rowIt->first, rowIt->second);
                ++rowIt;
            }

            return std::make_pair(it, true);
        }
    }

    /// @return Iterator on new allocated row
    /// Creates a new row in the sparse matrix with the last+1 key index
    RowIterator newLine()
    {
        KeyType newId = m_data.empty() ? 0 : (m_data.rbegin()->first + 1);

        std::pair< typename SparseMatrix::iterator, bool > res = m_data.insert(std::make_pair(newId, RowType()));
        return RowIterator(res.first);
    }
};



} // namespace defaulttype

} // namespace sofa

#endif // SOFA_DEFAULTTYPE_MAPMAPSPARSEMATRIX_H
