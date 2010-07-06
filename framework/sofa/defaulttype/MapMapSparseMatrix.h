
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_MAPMAPSPARSEMATRIX_H
#define SOFA_DEFAULTTYPE_MAPMAPSPARSEMATRIX_H

#include <map>

namespace sofa
{

namespace defaulttype
{

template <class T>
class MapMapSparseMatrix
{
public:
    typedef std::map< unsigned int, T > ColType;
    typedef std::map< unsigned int, ColType > SparseMatrix;
    typedef T Data;

    void clear()
    {
        m_data.clear();
    }

    bool empty() const { return m_data.empty(); }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const MapMapSparseMatrix<T>& /*sc*/ )
    {
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, MapMapSparseMatrix<T>& /*sc*/ )
    {
        return in;
    }

protected:
    SparseMatrix m_data;

public:

    class RowConstIterator
    {
    protected:
        RowConstIterator()
            : m_matrix(NULL)
        {

        }

        RowConstIterator(const SparseMatrix* _matrix, typename SparseMatrix::const_iterator _internal)
            : m_matrix(_matrix), m_internal(_internal)
        {

        }

    public:

        RowConstIterator(const RowConstIterator& it2)
            : m_matrix(it2.m_matrix), m_internal(it2.m_internal)
        {

        }

        void operator=(const RowConstIterator& it2)
        {
            m_matrix = it2.m_matrix;
            m_internal = it2.m_internal;
        }

        RowConstIterator begin()
        {
            return RowConstIterator(this->m_matrix, this->m_matrix->begin());
        }

        RowConstIterator end()
        {
            return RowConstIterator(this->m_matrix, this->m_matrix->end());
        }

        const std::pair< typename SparseMatrix::key_type, ColType >& operator*()
        {
            return *m_internal;
        }

        const std::pair< typename SparseMatrix::key_type, ColType >& operator->()
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

    private:
        const SparseMatrix* m_matrix;
        typename SparseMatrix::const_iterator m_internal;
    };

    class ColConstIterator
    {
    protected:
        ColConstIterator()
            : m_col(NULL)
        {

        }

        ColConstIterator(const ColType _col, typename ColType::const_iterator _internal, const unsigned int _rowIndex)
            : m_col(_col)
            , m_internal(_internal)
            , m_rowIndex(_rowIndex)
        {

        }

    public:

        ColConstIterator(const ColConstIterator& it2)
            : m_col(it2.m_col), m_internal(it2.m_internal), m_rowIndex(it2.m_rowIndex)
        {

        }

        void operator=(const ColConstIterator& it2)
        {
            m_col = it2.m_col;
            m_internal = it2.m_internal;
            m_rowIndex = it2.m_rowIndex;
        }

        unsigned int row()
        {
            return m_rowIndex;
        }

        ColConstIterator begin()
        {
            return ColConstIterator(this->m_col, this->m_col->begin(), this->row());
        }

        ColConstIterator end()
        {
            return ColConstIterator(this->m_col, this->m_col->end(), this->row());
        }

        const std::pair< typename ColType::key_type, T >& operator*()
        {
            return *m_internal;
        }

        const std::pair< typename ColType::key_type, T >& operator->()
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
        const ColType *m_col;
        typename ColType::const_iterator m_internal;
        const unsigned int m_rowIndex;
    };
};

} // namespace defaulttype

} // namespace sofa

#endif // SOFA_DEFAULTTYPE_MAPMAPSPARSEMATRIX_H
