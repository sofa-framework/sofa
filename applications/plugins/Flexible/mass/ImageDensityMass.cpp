#define SOFA_FLEXIBLE_ImageDensityMass_CPP

#include "ImageDensityMass.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;



////template <> SOFA_BASE_MECHANICS_API
//SReal ImageDensityMass<Rigid3Types,core::behavior::ShapeFunctiond,Rigid3Mass>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx ) const
//{
//    const VecCoord& _x = vx.getValue();

//    VecCoord Mx = m_massMatrix * _x;

//    SReal e = 0;
//    // gravity
//    Vec3d g ( this->getContext()->getGravity() );
//    for( unsigned int i=0 ; i<_x.size() ; i++ )
//    {
//        e -= g*Mx[i].getCenter();
//    }
//    return e;
//}


//
//#ifdef SOFA_WITH_FLOAT
//template <> SOFA_BASE_MECHANICS_API
//SReal ImageDensityMass<Rigid3fTypes,core::behavior::ShapeFunctionf,Rigid3fMass>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx ) const
//{
//    const VecCoord& _x = vx.getValue();

//    VecCoord Mx = m_massMatrix * _x;

//    float e = 0;
//    // gravity
//    Vec3f g ( this->getContext()->getGravity() );
//    for( unsigned int i=0 ; i<_x.size() ; i++ )
//    {
//        e -= g*Mx[i].getCenter();
//    }
//    return e;
//}


//#endif




// Register in the Factory
int ImageDensityMassClass = core::RegisterObject("Define a global mass matrix including non diagonal terms")
        .add< ImageDensityMass<Vec3Types,core::behavior::ShapeFunctiond,Mat3x3d> >( true )
//        .add< ImageDensityMass<Rigid3Types,core::behavior::ShapeFunctiond,Rigid3Mass> >()

        ;
template class SOFA_Flexible_API ImageDensityMass<Vec3Types,core::behavior::ShapeFunctiond,Mat3x3d>;
//template class SOFA_Flexible_API ImageDensityMass<Rigid3Types,core::behavior::ShapeFunctiond,Rigid3Mass>;





} // namespace mass

} // namespace component

} // namespace sofa

