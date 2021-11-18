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

#include <sofa/core/DataTracker.h>
#include <sofa/core/objectmodel/DDGNode.h>
namespace sofa::core
{

/// A DDGNode that will call a given Functor as soon as one of its input changes
/// (a pointer to this DataTrackerFunctor is passed as parameter in the functor)
template <typename FunctorType>
class DataTrackerFunctor : public core::objectmodel::DDGNode
{
public:

    DataTrackerFunctor( FunctorType& functor )
        : core::objectmodel::DDGNode()
        , m_functor( functor )
    {}

    /// The trick is here, this function is called as soon as the input data changes
    /// and can then trigger the callback
    void setDirtyValue() override
    {
        m_functor( this );

        // the input needs to be inform their output (including this DataTrackerFunctor)
        // are not dirty, to be sure they will call setDirtyValue when they are modified
        cleanDirtyOutputsOfInputs();
    }


    /// This method is needed by DDGNode
    void update() override{}

private:

    DataTrackerFunctor(const DataTrackerFunctor&);
    void operator=(const DataTrackerFunctor&);
    FunctorType& m_functor; ///< the functor to call when the input data changed

};

} // namespace sofa::core

