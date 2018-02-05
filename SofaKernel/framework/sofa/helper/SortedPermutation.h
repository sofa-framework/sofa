/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_SortedPermutation_H
#define SOFA_HELPER_SortedPermutation_H


/** Utility to compute the sorted permutation of a container. See example at the end of the file
  Francois Faure, April 2012
  */

#include <sofa/helper/vector.h>
#include <iostream>
#include <algorithm>

namespace sofa
{

namespace helper
{

/** Comparison operator used to compute sorted permutations of a container.
  The comparison operator of two indices compares the corresponding entries of the container.
  The container must allow random access.
  */
template<class Container>
struct CompareIndirect
{
    const Container& values;
    CompareIndirect( const Container& v ):values(v) {}

    bool operator () (unsigned i, unsigned j) const { return values[i] < values[j]; }
};

/// Return a sorted permutation of the container, i.e. a list of indices corresponding to increasing entries.
template<class Container>
helper::vector<unsigned> sortedPermutation( const Container& values )
{
    helper::vector<unsigned> permutation;
    permutation.resize(values.size());
    for(unsigned i=0; i<permutation.size(); i++)
        permutation[i] = i;

    CompareIndirect<Container> cmp(values);
    std::sort( permutation.begin(), permutation.end(), cmp );
    return permutation;

}

////Example:
///    helper::vector<double> values;
///     values.push_back(24);
///     values.push_back(55);
///     values.push_back(22);
///     values.push_back(1);
///     vector<unsigned> permutation = sortedPermutation(values);
///    //The following prints: 3 2 0 1
///     std::cout << permutation << "\n";



} // namespace helper

} // namespace sofa

#endif
