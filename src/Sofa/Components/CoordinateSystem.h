#ifndef SOFA_COMPONENTS_CoordinateSystem_H
#define SOFA_COMPONENTS_CoordinateSystem_H

#include <Sofa/Core/Property.h>
#include <Sofa/Components/Common/RigidTypes.h>

namespace Sofa
{

namespace Components
{

class CoordinateSystem : public Core::Property
{
public:
    typedef Sofa::Components::Common::RigidTypes::Coord Frame;
    const Frame&  getFrame() const { return frame_; }
    void setFrame( const Frame& f ) { frame_=f; }

    void updateProperties( Core::Properties& data )
    {
        data.worldTransform.multRight(frame_);
    }
protected:
    Frame frame_;
};

} // namespace Components

} // namespace Sofa

#endif


