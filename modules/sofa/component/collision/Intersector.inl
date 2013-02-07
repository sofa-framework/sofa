// File modified from GeometricTools
// http://www.geometrictools.com/

#include <sofa/component/collision/Intersector.h>

namespace sofa{

namespace component{

namespace collision{

using namespace sofa::defaulttype;

//----------------------------------------------------------------------------
template <class TDataTypes>
Intersector<TDataTypes>::Intersector ()
{
    mContactTime = (Real)0;
    mIntersectionType = IT_EMPTY;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
Intersector<TDataTypes>::~Intersector ()
{
}
//----------------------------------------------------------------------------
template <class TDataTypes>
typename Intersector<TDataTypes>::Real Intersector<TDataTypes>::GetContactTime () const
{
    return mContactTime;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
int Intersector<TDataTypes>::GetIntersectionType () const
{
    return mIntersectionType;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool Intersector<TDataTypes>::Test ()
{
    // Stub for derived class.
    //assertion(false, "Function not yet implemented\n");
    return false;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool Intersector<TDataTypes>::Find ()
{
    // Stub for derived class.
    //assertion(false, "Function not yet implemented\n");
    return false;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool Intersector<TDataTypes>::Test (Real, const TVector&, const TVector&)
{
    // Stub for derived class.
    //assertion(false, "Function not yet implemented\n");
    return false;
}
//----------------------------------------------------------------------------
template <class TDataTypes>
bool Intersector<TDataTypes>::Find (Real, const TVector&, const TVector&)
{
    // Stub for derived class.
    //assertion(false, "Function not yet implemented\n");
    return false;
}
//----------------------------------------------------------------------------

}
}
}
