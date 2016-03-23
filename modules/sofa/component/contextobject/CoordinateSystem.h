/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONTEXTOBJECT_COORDINATESYSTEM_H
#define SOFA_COMPONENT_CONTEXTOBJECT_COORDINATESYSTEM_H
// Author: François Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{
namespace core
{
namespace objectmodel
{
class Context;
}

}
namespace simulation
{
class Node;
}

namespace component
{

namespace contextobject
{

/** Defines the local coordinate system with respect to its parent.
*/
class CoordinateSystem : public core::objectmodel::ContextObject
{
public:
    SOFA_CLASS(CoordinateSystem, core::objectmodel::ContextObject);
    typedef core::objectmodel::BaseContext::SolidTypes SolidTypes;
    typedef SolidTypes::Vec Vec;
    typedef SolidTypes::Rot Rot;
    typedef SolidTypes::Mat Mat;
    typedef SolidTypes::Coord Frame;
    typedef SolidTypes::Deriv SpatialVector;

    CoordinateSystem();
    virtual ~CoordinateSystem()
    {}

    // ContextObject
    virtual void apply();

    virtual void draw(const core::visual::VisualParams* vparams);

    virtual void reinit();

    virtual void init();

    /// Transform wrt parent
    const Frame&  getTransform() const;
    /// Transform wrt parent
    virtual void setTransform( const Frame& f );
    /// Transform wrt world
    //Frame  getPositionInWorld() const;
    /// Define translation in parent coordinates
    //CoordinateSystem* setOrigin( const Vec& t );
    /// Translation in parent coordinates
    Vec getOrigin() const;
    /// Define orientation (rotation of the child wrt parent)  in parent coordinates
    //CoordinateSystem* setOrientation( const Rot& r );
    /// Orientation (rotation of the child wrt parent)
    Rot getOrientation() const;
    /// Define translation and orientation  in parent coordinates
    //CoordinateSystem* set( const Vec& t, const Rot& r );

    /// wrt world, given in world coordinates
    //SpatialVector  getVelocityInWorld() const;
    /// wrt parent frame, given in LOCAL frame
    //const SpatialVector&  getVelocity() const;
    /// wrt parent frame, given in LOCAL frame
    //CoordinateSystem* setVelocity( const SpatialVector& f );

    Data< defaulttype::Vec3f > origin;
    Data< defaulttype::Vec3f > orientation;
protected:
    Frame positionInParent_;   ///< wrt parent frame
    //SpatialVector velocity_;  ///< velocity wrt parent frame, given in LOCAL frame


};

} // namespace contextobject

} // namespace component

} // namespace sofa

#endif


