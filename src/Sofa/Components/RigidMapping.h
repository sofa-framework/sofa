#ifndef SOFA_COMPONENTS_RIGIDMAPPING_H
#define SOFA_COMPONENTS_RIGIDMAPPING_H

#include "Sofa/Core/MechanicalMapping.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Components/RigidObject.h"
#include "Sofa/Abstract/VisualModel.h"
#include <vector>

namespace Sofa
{

namespace Components
{

template <class BaseMapping>
class RigidMapping : public BaseMapping, public Abstract::VisualModel
{
public:
    typedef BaseMapping Inherit;
    typedef typename BaseMapping::In In;
    typedef typename BaseMapping::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Coord::value_type Real;

    class Mat3 : public fixed_array<Deriv,3>
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
    void init(const char* filename);
public:
    RigidMapping(In* from, Out* to, const char* filename)
        : Inherit(from, to)
    {
        init(filename);
    }

    RigidMapping(In* from, Out* to)
        : Inherit(from, to)
    {
    }

    virtual ~RigidMapping()
    {
    }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
