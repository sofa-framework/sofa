#ifndef SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/core/VisualModel.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BaseMapping>
class LaparoscopicRigidMapping : public BaseMapping, public core::VisualModel
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
    defaulttype::Vector3 pivot;
    defaulttype::Quat rotation;
public:

    LaparoscopicRigidMapping(In* from, Out* to)
        : Inherit(from, to)
    {
    }

    virtual ~LaparoscopicRigidMapping()
    {
    }

    void setPivot(const defaulttype::Vector3& val) { this->pivot = val; }
    void setRotation(const defaulttype::Quat& val) { this->rotation = val; this->rotation.normalize(); }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

protected:

    bool getShow(const core::objectmodel::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const core::componentmodel::behavior::BaseMechanicalMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
