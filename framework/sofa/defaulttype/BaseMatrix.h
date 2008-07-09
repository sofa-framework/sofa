/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_DEFAULTTYPE_BASEMATRIX_H
#define SOFA_DEFAULTTYPE_BASEMATRIX_H

namespace sofa
{

namespace defaulttype
{

/// Generic matrix API, allowing to fill and use a matrix independently of the linear algebra library in use.
///
/// Note that accessing values using this class is rather slow and should only be used in codes where the
/// provided genericity is necessary.
class BaseMatrix
{
public:
    virtual ~BaseMatrix() {}

    /// Number of rows
    virtual int rowSize(void) const = 0;
    /// Number of columns
    virtual int colSize(void) const = 0;
    /// Read the value of the element at row i, column j (using 0-based indices)
    virtual SReal element(int i, int j) const = 0;
    /// Resize the matrix and reset all values to 0
    virtual void resize(int nbRow, int nbCol) = 0;
    /// Reset all values to 0
    virtual void clear() = 0;
    /// Write the value of the element at row i, column j (using 0-based indices)
    virtual void set(int i, int j, double v) = 0;
    /// Add v to the existing value of the element at row i, column j (using 0-based indices)
    virtual void add(int i, int j, double v) = 0;
    /*    /// Write the value of the element at row i, column j (using 0-based indices)
        virtual void set(int i, int j, float v) { set(i,j,(double)v); }
        /// Add v to the existing value of the element at row i, column j (using 0-based indices)
        virtual void add(int i, int j, float v) { add(i,j,(double)v); }
        /// Reset the value of element i,j to 0
    */    virtual void clear(int i, int j) { set(i,j,0.0); }
    /// Reset the value of row i to 0
    virtual void clearRow(int i) { for (int j=0,n=colSize(); j<n; ++j) clear(i,j); }
    /// Reset the value of column j to 0
    virtual void clearCol(int j) { for (int i=0,n=rowSize(); i<n; ++i) clear(i,j); }
    /// Reset the value of both row and column i to 0
    virtual void clearRowCol(int i) { clearRow(i); clearCol(i); }
};


} // nampespace defaulttype

} // nampespace sofa

#endif
