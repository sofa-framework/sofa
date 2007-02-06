#ifndef SOFA_COMPONENTS_COORDINATESYSTEM_H
#define SOFA_COMPONENTS_COORDINATESYSTEM_H
// Author: François Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#include <Sofa-old/Abstract/ContextObject.h>
#include <Sofa-old/Abstract/VisualModel.h>
#include <Sofa-old/Components/Common/SolidTypes.h>

namespace Sofa
{
namespace Core
{
class Context;
class Topology;
}

namespace Components
{
namespace Graph
{
class GNode;
}


/** Defines the local coordinate system with respect to its parent.
*/
class CoordinateSystem : public Abstract::ContextObject, public Abstract::VisualModel
{
public:
    typedef Abstract::BaseContext::SolidTypes SolidTypes;
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

    // VisualModel
    virtual void draw();
    virtual void initTextures()
    {}
    virtual void update()
    {}


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

protected:
    Frame positionInParent_;   ///< wrt parent frame
    //SpatialVector velocity_;  ///< velocity wrt parent frame, given in LOCAL frame


};

} // namespace Components

} // namespace Sofa

#endif


