#ifndef SOFA_COMPONENTS_LAPAROSCOPICRIGIDMAPPING_H
#define SOFA_COMPONENTS_LAPAROSCOPICRIGIDMAPPING_H

#include "Sofa/Core/MechanicalMapping.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Common/RigidTypes.h"
#include "Common/LaparoscopicRigidTypes.h"
#include "Sofa/Abstract/VisualModel.h"
#include <vector>

namespace Sofa
{

namespace Components
{

template <class BaseMapping>
class LaparoscopicRigidMapping : public BaseMapping, public Abstract::VisualModel
{
public:
    typedef BaseMapping Inherit;
    typedef typename BaseMapping::In In;
    typedef typename BaseMapping::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    //typedef typename Coord::value_type Real;

protected:
    Common::Vector3 pivot;
    Common::Quat rotation;
public:

    LaparoscopicRigidMapping(In* from, Out* to)
        : Inherit(from, to)
    {
    }

    virtual ~LaparoscopicRigidMapping()
    {
    }

    void setPivot(const Common::Vector3& val) { this->pivot = val; }
    void setRotation(const Common::Quat& val) { this->rotation = val; this->rotation.normalize(); }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

protected:

    bool getShow(const Abstract::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const Core::BasicMechanicalMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }
};

} // namespace Components

} // namespace Sofa

#endif
