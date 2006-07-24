#ifndef SOFA_COMPONENTS_COORDINATESYSTEM_H
#define SOFA_COMPONENTS_COORDINATESYSTEM_H

#include <Sofa/Abstract/ContextObject.h>
#include <Sofa/Abstract/VisualModel.h>
#include <Sofa/Components/Common/SolidTypes.h>
// #include <Sofa/Components/Common/vector.h>

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
    virtual void initTextures() {}
    virtual void update() {}


    const Frame&  getRelativePosition() const;
    CoordinateSystem* setRelativePosition( const Frame& f );
//     CoordinateSystem* setFrame( const Vec& translation, const Rot& rotation );
//     CoordinateSystem* setFrame( const Vec& translation ) { Rot r = Rot::identity(); return setFrame(translation, r); }

    /// wrt parent frame, given in parent frame
    const SpatialVector&  getRelativeVelocity() const;
    /// wrt parent frame, given in parent frame
    CoordinateSystem* setRelativeVelocity( const SpatialVector& f );
    /*    const Vec& getLinearVelocity() const;
        CoordinateSystem* setLinearVelocity( const Vec& linearVelocity);
        const Vec& getAngularVelocity() const;
        CoordinateSystem* setAngularVelocity( const Vec& angularVelocity );*/

protected:
    Frame relativePosition_;   ///< wrt parent frame
    SpatialVector relativeVelocity_;  ///< wrt parent frame, given in parent frame


};

} // namespace Components

} // namespace Sofa

#endif
