/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
//
// C++ Interface: TriangleBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_H
#include "config.h"

#include <SofaDeformable/StiffSpringForceField.h>
#include <map>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/**
Bending springs added between vertices of triangles sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.


	@author The SOFA team </www.sofa-framework.org>
 */
template<class DataTypes>
class TriangleBendingSprings : public sofa::component::interactionforcefield::StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangleBendingSprings, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
protected:
    TriangleBendingSprings();

    ~TriangleBendingSprings();
public:
    /// Searches triangle topology and creates the bending springs
    virtual void init();

    //virtual void draw()
    //{
    //}

protected:
    typedef std::pair<unsigned,unsigned> IndexPair;
    void addSpring( unsigned, unsigned );
    void registerTriangle( unsigned, unsigned, unsigned, std::map<IndexPair, unsigned>& );

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<defaulttype::Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_DEFORMABLE_API TriangleBendingSprings<defaulttype::Vec2fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_TRIANGLEBENDINGSPRINGS_H */
