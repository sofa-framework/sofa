#include "MassObject1d.h"

#include "Sofa/Core/MechanicalMapping.inl"
#include "Sofa/Components/XML/MappingNode.h"

#include <math.h>

using namespace Sofa::Components;
using namespace Sofa::Components::Common;
using namespace Sofa::Core;

class CircleMapping : public Sofa::Core::MechanicalMapping< Sofa::Core::MechanicalObject<Vec1dTypes>, Sofa::Core::MechanicalObject<Vec3dTypes> >
{
public:
    // Simplified notation for all involved classes
    typedef Sofa::Core::MechanicalMapping< Sofa::Core::MechanicalObject<Vec1dTypes>, Sofa::Core::MechanicalObject<Vec3dTypes> > BaseMapping;
    typedef BaseMapping::In In;
    typedef BaseMapping::Out Out;
    typedef Out::VecCoord VecCoord;
    typedef Out::VecDeriv VecDeriv;
    typedef Out::Coord Coord;
    typedef Out::Deriv Deriv;

    Coord p0; ///< Origin of the circle
    Deriv rx, ry; ///< Radius of the circle

    std::vector<Deriv> dx;

    CircleMapping(In* from, Out* to)
        : BaseMapping(from, to), p0(0,0,0), rx(1,0,0), ry(0,0,1)
    {
    }

    void apply( Out::VecCoord& out, const In::VecCoord& in )
    {
        out.resize(in.size());
        dx.resize(in.size());
        for(unsigned int i=0; i<out.size(); i++)
        {
            double c = cos(in[i]);
            double s = sin(in[i]);
            out[i] = p0+rx*c+ry*s;
            dx[i] = rx*(-s)+ry*c;
        }
    }

    void applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
    {
        out.resize(in.size());
        for(unsigned int i=0; i<out.size(); i++)
            out[i] = dx[i]*in[i];
    }

    void applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
    {
        for(unsigned int i=0; i<out.size(); i++)
            out[i] += dx[i]*in[i];
    }
};


namespace Sofa
{
namespace Components
{
namespace Common
{

void create(CircleMapping*& obj, XML::Node<Sofa::Core::BasicMapping>* arg)
{
    XML::createWith2Objects< CircleMapping, CircleMapping::In, CircleMapping::Out>(obj, arg);
    if (obj!=NULL)
    {
        double r = atof(arg->getAttribute("r","1"));
        obj->p0[0] = atof(arg->getAttribute("x0","0"));
        obj->p0[1] = atof(arg->getAttribute("y0","0"));
        obj->p0[2] = atof(arg->getAttribute("z0","0"));
        obj->rx[0] = atof(arg->getAttribute("rxx","1"))*r;
        obj->rx[1] = atof(arg->getAttribute("rxy","0"))*r;
        obj->rx[2] = atof(arg->getAttribute("rxz","0"))*r;
        obj->ry[0] = atof(arg->getAttribute("ryx","0"))*r;
        obj->ry[1] = atof(arg->getAttribute("ryy","0"))*r;
        obj->ry[2] = atof(arg->getAttribute("ryz","1"))*r;
    }
}
}
}
}

Creator< XML::MappingNode::Factory, CircleMapping > CircleMappingClass("CircleMapping", true);
