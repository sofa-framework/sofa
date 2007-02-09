#ifndef SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/VisualModel.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class RigidMapping : public BasicMapping, public core::VisualModel
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Coord::value_type Real;

    class Mat3 : public helper::fixed_array<Deriv,3>
    {
    public:
        Coord operator*(const Coord& v) const
        {
            return Coord((*this)[0]*v,(*this)[1]*v,(*this)[2]*v);
        }
    };

protected:
    std::vector<Coord> points;
    Coord translation;
    Real orientation[4];
    Mat3 rotation;
    std::vector<Coord> rotatedPoints;
    class Loader;
    void load(const char* filename);
public:

    RigidMapping(In* from, Out* to)
        : Inherit(from, to)
    {
    }

    virtual ~RigidMapping()
    {
    }

    void init();

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("filename"))
            this->load(arg->getAttribute("filename"));
        this->Inherit::parse(arg);
    }

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

    //virtual const char* getTypeName() const { return "RigidMapping"; }

protected:

    bool getShow(const core::objectmodel::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const core::componentmodel::behavior::BaseMechanicalMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
