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
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/helper/logging/Messaging.h>
#include <climits>

namespace sofa::linearalgebra
{

BaseMatrix::BaseMatrix() {}

BaseMatrix::~BaseMatrix()
{}

void BaseMatrix::compress() {}

///Adding values from a 3x3d matrix this function may be overload to obtain better performances
void BaseMatrix::add(Index row, Index col, const type::Mat3x3d & _M) {
    for (unsigned i=0;i<3;i++)
        for (unsigned j=0;j<3;j++)
            add(row + i, col + j, _M[i][j]);
}

///Adding values from a 3x3f matrix this function may be overload to obtain better performances
void BaseMatrix::add(Index row, Index col, const type::Mat3x3f & _M) {
    for (unsigned i=0;i<3;i++)
        for (unsigned j=0;j<3;j++)
            add(row + i, col + j, _M[i][j]);
}

///Adding values from a 2x2d matrix this function may be overload to obtain better performances
void BaseMatrix::add(Index row, Index col, const type::Mat2x2d & _M) {
    for (unsigned i=0;i<2;i++)
        for (unsigned j=0;j<2;j++)
            add(row + i, col + j, _M[i][j]);
}

///Adding values from a 2x2f matrix this function may be overload to obtain better performances
void BaseMatrix::add(Index row, Index col, const type::Mat2x2f & _M) {
    for (unsigned i=0;i<2;i++)
        for (unsigned j=0;j<2;j++)
            add(row + i, col + j, _M[i][j]);
}

std::ostream& operator<<(std::ostream& out, const  sofa::linearalgebra::BaseMatrix& m )
{
    Index nx = m.colSize();
    Index ny = m.rowSize();
    out << "[";
    for (Index y=0; y<ny; ++y)
    {
        out << "\n[";
        for (Index x=0; x<nx; ++x)
        {
            out << " " << m.element(y,x);
        }
        out << " ]";
    }
    out << " ]";
    return out;
}

std::istream& operator>>( std::istream& in, sofa::linearalgebra::BaseMatrix& m )
{
    // The reading could be way simplier with an other format,
    // but I did not want to change the existing output.
    // Anyway, I guess there are better ways to perform the reading
    // but at least this one is working...

    std::vector<SReal> line;
    std::vector< std::vector<SReal> > lines;

//    unsigned l=0, c;

    in.ignore(INT_MAX, '['); // ignores all characters until it passes a [, start of the matrix

    while(true)
    {
        in.ignore(INT_MAX, '['); // ignores all characters until it passes a [, start of the line
//        c=0;

        SReal r;
        char car; in >> car;
        while( car!=']') // end of the line
        {
            in.seekg( -1, std::istream::cur ); // unread car
            in >> r;
            line.push_back(r);
//            ++c;
            in >> car;
        }

//        ++l;

        lines.push_back(line);
        line.clear();

        in >> car;
        if( car==']' ) break; // end of the matrix
        else in.seekg( -1, std::istream::cur ); // unread car

    }

    m.resize( (Index)lines.size(), (Index)lines[0].size() );

    for( size_t i=0; i<lines.size();++i)
    {
        assert( lines[i].size() == lines[0].size() ); // all line should have the same number of columns
        for( size_t j=0; j<lines[i].size();++j)
        {
            m.add( (Index)i, (Index)j, lines[i][j] );
        }
    }

    m.compress();


    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}


} // namespace sofa::linearalgebra
