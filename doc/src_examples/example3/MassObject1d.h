#include "Sofa/Components/MassObject.h"
#include "Sofa/Components/Common/VecTypes.h"

using namespace Sofa::Components;
using namespace Sofa::Components::Common;
using namespace Sofa::Core;

class Vec1dTypes
{
public:
    typedef double Coord;
    typedef double Deriv;
    typedef std::vector<Coord> VecCoord;
    typedef std::vector<Deriv> VecDeriv;

    static void set(Coord& c, double x, double /*y*/, double /*z*/)
    {
        c = x;
    }

    static void add(Coord& c, double x, double /*y*/, double /*z*/)
    {
        c += x;
    }
};

typedef MassObject<Vec1dTypes> MassObject1d;
