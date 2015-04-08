#include "ComputeDualQuatEngine.h"

#include "../types/AffineTypes.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/DualQuat.h>

namespace
{

    
    // Does nothing
    template < typename Real >
    const sofa::defaulttype::RigidCoord< 3, Real >& getRigid(  const typename sofa::defaulttype::StdRigidTypes< 3, Real >::Coord& x )
    {
        return x;
    }


    // Transforms an Affine to a Rigid
    template < typename Real >
    const sofa::defaulttype::RigidCoord< 3, Real > getRigid(  typename sofa::defaulttype::StdAffineTypes< 3, Real >::Coord x)
    {
        typedef sofa::defaulttype::RigidCoord< 3, Real > Rigid;
        typedef sofa::helper::Quater<Real> Quater;

        // Project the Affine to a rigid (is that necessary ?)
        x.setRigid();

        // Extract the Rigid from the affine
        Quater r;
        r.fromMatrix( x.getAffine() );    

        return Rigid( x.getCenter(), r ) ;
    }

}


namespace sofa
{

namespace component
{

namespace engine
{

template < class DataTypes >
ComputeDualQuatEngine< DataTypes >::ComputeDualQuatEngine()
    : d_x0( initData( &d_x0, "x0", "Rest position" ) )
    , d_x( initData( &d_x, "x", "Current position" ) )
    , d_dualQuats( initData( &d_dualQuats, "dualQuats", "Dual quaternions, computed from x (or x*x0^-1 if x0 is provided). DualQuats are stored as two vec4f elements, first the orientation, then the dual.") )
{
    addInput( &d_x0 );
    addInput( &d_x );
    addOutput( &d_dualQuats );
    setDirtyValue();
}


template < class DataTypes >
void ComputeDualQuatEngine< DataTypes >::update()
{
    cleanDirty();
    typedef sofa::helper::DualQuatCoord3<Real> DualQuat;

    // Clean the output
    helper::vector< defaulttype::Vec4f >& dualQuats = *d_dualQuats.beginEdit();
    dualQuats.clear();

    const VecCoord& x = d_x.getValue();
    const VecCoord& x0 = d_x0.getValue();
    const size_t size = x.size();
    const size_t size0 = x0.size();

    // Check the size of x0
    if( size0 != 0 && size != size0 )
    {
        serr << "x and x0 have not the same size: respectively " << size << " and " << size0 << sendl;
        return;
    }


    for( unsigned int i = 0; i < size; ++i )
    {
        typedef sofa::defaulttype::RigidCoord< 3, Real > Rigid;
        // Transform to rigid (if necessary)
        Rigid p = getRigid<Real>( x[i] );

        if( size0 != 0 )
        {
            Rigid p0 = getRigid<Real>( x0[i] );
            // Compute X0^-1*X
            p = p.mult( sofa::defaulttype::StdRigidTypes< 3, Real >::inverse(p0) );
        }

        // Pass to a dualquat
        DualQuat dualQuat( p );

        // Insert into the output
        sofa::defaulttype::Vec<4,Real> orientation = dualQuat.getOrientation();
        sofa::defaulttype::Vec<4,Real> dual = dualQuat.getDual();
        dualQuats.push_back( defaulttype::Vec4f( orientation[0], orientation[1], orientation[2], orientation[3] ) );
        dualQuats.push_back( defaulttype::Vec4f( dual[0], dual[1], dual[2], dual[3] ) );
    }

    d_dualQuats.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa
