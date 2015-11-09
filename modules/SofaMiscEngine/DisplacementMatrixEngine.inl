#include "DisplacementMatrixEngine.h"

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/DualQuat.h>
#include <iostream>
using std::cerr;
using std::endl;

//namespace
//{

    
//    // Does nothing
//    template < typename Real >
//    const sofa::defaulttype::RigidCoord< 3, Real >& getRigid(  const typename sofa::defaulttype::StdRigidTypes< 3, Real >::Coord& x )
//    {
//        return x;
//    }


//    // Transforms an Affine to a Rigid
//    template < typename Real >
//    const sofa::defaulttype::RigidCoord< 3, Real > getRigid(  typename sofa::defaulttype::StdAffineTypes< 3, Real >::Coord x)
//    {
//        typedef sofa::defaulttype::RigidCoord< 3, Real > Rigid;
//        typedef sofa::helper::Quater<Real> Quater;

//        // Project the Affine to a rigid (is that necessary ?)
//        x.setRigid();

//        // Extract the Rigid from the affine
//        Quater r;
//        r.fromMatrix( x.getAffine() );

//        return Rigid( x.getCenter(), r ) ;
//    }

//}


namespace sofa
{

namespace component
{

namespace engine
{

template < class DataTypes >
DisplacementMatrixEngine< DataTypes >::DisplacementMatrixEngine()
    : d_x0( initData( &d_x0, "x0", "Rest position" ) )
    , d_x( initData( &d_x, "x", "Current position" ) )
    , d_displaceMats( initData( &d_displaceMats, "displaceMats", "Displacement matrices with respect to original rigid positions") )
{
    addInput( &d_x0 );
    addInput( &d_x );
    addOutput( &d_displaceMats );
    setDirtyValue();
}


template < class DataTypes >
void DisplacementMatrixEngine< DataTypes >::update()
{
    typedef sofa::helper::DualQuatCoord3<Real> DualQuat;


    const VecCoord& x = d_x.getValue();
    const VecCoord& x0 = d_x0.getValue();
    const size_t size = x.size();
    const size_t size0 = x0.size();

    // Check the size of x0
    if( size != size0 )
    {
        serr << "x and x0 have not the same size: respectively " << size << " and " << size0 << sendl;
        return;
    }

    cleanDirty();

    // Clean the output
    helper::vector< defaulttype::Mat4x4f >& displaceMats = *d_displaceMats.beginWriteOnly();
    displaceMats.resize(size);

    for( unsigned int i = 0; i < size; ++i )
    {
        x[i].toMatrix(displaceMats[i]);
//        cerr << "DisplacementMatrixEngine< DataTypes >::update(), x[i]  = " << x[i] << endl;
//        cerr << "DisplacementMatrixEngine< DataTypes >::update(), x0[i] = " << x0[i] << endl;
        Mat4 inv;
        sofa::defaulttype::StdRigidTypes< 3, Real >::inverse(x0[i]).toMatrix(inv);
//        cerr << "DisplacementMatrixEngine< DataTypes >::update(), mat   = " << displaceMats[i] << endl;
        displaceMats[i] = displaceMats[i] * inv;
//        cerr << "DisplacementMatrixEngine< DataTypes >::update(), inv   = " << inv << endl;
//        cerr << "DisplacementMatrixEngine< DataTypes >::update(), disp  = " << displaceMats[i] << endl;
    }

    d_displaceMats.endEdit();
//    cerr << "DisplacementMatrixEngine< DataTypes >::update(), displaceMats  = " << d_displaceMats.getValue() << endl;
}

} // namespace engine

} // namespace component

} // namespace sofa
