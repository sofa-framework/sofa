#include "CoordinateSystem.h"

namespace Sofa
{

namespace Components
{



CoordinateSystem::CoordinateSystem(std::string name, Graph::GNode* n):Abstract::ContextObject(name,n)
{
    frame_.clear();
    n->addObject(this);
}


const CoordinateSystem::Frame&  CoordinateSystem::getFrame() const
{
    return frame_;
}
void CoordinateSystem::setFrame( const CoordinateSystem::Frame& f )
{
    frame_=f;
}
void CoordinateSystem::setFrame( const CoordinateSystem::Vec3& translation, const CoordinateSystem::Quat& rotation, const CoordinateSystem::Vec3& scale )
{
    frame_=Frame(translation,rotation,scale);
}

void CoordinateSystem::apply()
{
    //cerr<<"CoordinateSystem, frame = "<<   frame_ << endl;
    getContext()->setLocalToWorld( getContext()->getLocalToWorld().mult( frame_ ) );
    getContext()->setWorldToLocal( getContext()->getLocalToWorld().inversed() );
}

} // namespace Components

} // namespace Sofa


