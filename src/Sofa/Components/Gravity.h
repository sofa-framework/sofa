#ifndef SOFA_COMPONENTS_GRAVITY_H
#define SOFA_COMPONENTS_GRAVITY_H

#include <Sofa/Components/Common/Vec.h>
#include <Sofa/Components/Graph/GNode.h>

namespace Sofa
{

namespace Components
{
using Common::Vec3f;

class Gravity : public Components::Graph::ContextObject
{
    typedef Common::Vec3f Vec3;
public:
    Gravity( std::string name, Components::Graph::GNode* gn ):ContextObject(name,gn) { gn->addObject(this); }
    const Vec3&  getGravity() const { return gravity_; }
    void setGravity( const Vec3& g ) { gravity_=g; }

    void apply() { getContext()->setGravity( gravity_ ); }
protected:
    Vec3 gravity_;
};

} // namespace Components

} // namespace Sofa

#endif


