#ifndef SOFA_COMPONENTS_FrameinateSystem_H
#define SOFA_COMPONENTS_CoordinateSystem_H

#include <Sofa/Abstract/BaseObject.h>
#include <Sofa/Components/Common/RigidTypes.h>
#include <Sofa/Components/Graph/GNode.h>
#include <Sofa/Components/Common/Frame.h>

namespace Sofa
{

namespace Components
{



class CoordinateSystem : public Abstract::ContextObject
{
public:
    typedef Common::Frame Frame;
    typedef Common::Vec3f Vec3;
    typedef Common::Quatf Quat;

    virtual void apply();
    CoordinateSystem(std::string name, Graph::GNode* n);
    virtual ~CoordinateSystem() {}


    const Frame&  getFrame() const;
    void setFrame( const Frame& f );
    void setFrame( const Vec3& translation, const Quat& rotation, const Vec3& scale=Vec3(1,1,1) );

protected:
    Frame frame_;
};

} // namespace Components

} // namespace Sofa

#endif

