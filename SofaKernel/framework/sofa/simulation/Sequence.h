/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_SIMULATION_SEQUENCE
#define SOFA_SIMULATION_SEQUENCE

#include <sofa/core/objectmodel/Link.h>
#include <sofa/simulation/Node_fwd.h>

namespace sofa {
namespace simulation {

///
///  class to hold a list of objects. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end).
/// UPDATE: it is now an alias for the Link pointer container
template < class T, bool strong = false >
class Sequence : public MultiLink<Node, T, BaseLink::FLAG_DOUBLELINK|(strong ? BaseLink::FLAG_STRONGLINK : BaseLink::FLAG_DUPLICATE)>
{
    public:
    typedef MultiLink<Node, T, BaseLink::FLAG_DOUBLELINK|(strong ? BaseLink::FLAG_STRONGLINK : BaseLink::FLAG_DUPLICATE)> Inherit;
    typedef T pointed_type;
    typedef typename Inherit::DestPtr value_type;
    //typedef TPtr value_type;
    typedef typename Inherit::const_iterator const_iterator;
    typedef typename Inherit::const_reverse_iterator const_reverse_iterator;
    typedef const_iterator iterator;
    typedef const_reverse_iterator reverse_iterator;

    Sequence(const BaseLink::InitLink<Node>& init)
        : Inherit(init)
    {
    }

    value_type operator[](unsigned int i) const
    {
        return this->get(i);
    }

    /// Swap two values in the list. Uses a const_cast to violate the read-only iterators.
    void swap( iterator a, iterator b )
    {
        value_type& wa = const_cast<value_type&>(*a);
        value_type& wb = const_cast<value_type&>(*b);
        value_type tmp = *a;
        wa = *b;
        wb = tmp;
    }
};

/// Class to hold 0-or-1 object. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end), plus an automatic convertion to one pointer.
/// UPDATE: it is now an alias for the Link pointer container
template < class T, bool duplicate = true >
class Single : public SingleLink<Node, T, BaseLink::FLAG_DOUBLELINK|(duplicate ? BaseLink::FLAG_DUPLICATE : BaseLink::FLAG_NONE)>
{
    public:
    typedef SingleLink<Node, T, BaseLink::FLAG_DOUBLELINK|(duplicate ? BaseLink::FLAG_DUPLICATE : BaseLink::FLAG_NONE)> Inherit;
    typedef T pointed_type;
    typedef typename Inherit::DestPtr value_type;
    //typedef TPtr value_type;
    typedef typename Inherit::const_iterator const_iterator;
    typedef typename Inherit::const_reverse_iterator const_reverse_iterator;
    typedef const_iterator iterator;
    typedef const_reverse_iterator reverse_iterator;

    Single(const BaseLink::InitLink<Node>& init)
        : Inherit(init)
    {
    }

    T* operator->() const
    {
        return this->get();
    }
    T& operator*() const
    {
        return *this->get();
    }
    operator T*() const
    {
        return this->get();
    }
};

} ///namespace simulation
} ///namespace sofa

#endif /// SOFA_SIMULATION_SEQUENCE
