#ifndef FLEXIBLE_AffineComponents_H
#define FLEXIBLE_AffineComponents_H


#include "AffineTypes.h"

#include <sofa/component/container/MechanicalObject.h>


#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/core/behavior/ForceField.h>



namespace sofa
{


// ==========================================================================
// Mechanical Object

namespace component
{
namespace container
{


template <> SOFA_Flexible_API
void MechanicalObject<defaulttype::Affine3Types>::draw(const core::visual::VisualParams* vparams);

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_AffineComponents_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Affine3dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Affine3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Affine3fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Affine3fTypes>;
#endif
#endif


} // namespace container







// ==========================================================================
// Uniform Mass


namespace mass
{

template<int N, typename Real>
class AddMToMatrixFunctor< typename defaulttype::StdAffineTypes<N,Real>::Deriv, defaulttype::AffineMass<N,Real> >
{
public:
    void operator()(defaulttype::BaseMatrix * mat, const defaulttype::AffineMass<N,Real>& mass, int pos, double fact)
    {
        typedef defaulttype::AffineMass<N,Real> AffineMass;
        for( unsigned i=0; i<AffineMass::VSize; ++i )
            for( unsigned j=0; j<AffineMass::VSize; ++j )
            {
                mat->add(pos+i, pos+j, mass[i][j]*fact);
//            cerr<<"AddMToMatrixFunctor< defaulttype::Vec<N,Real>, defaulttype::Mat<N,N,Real> >::operator(), add "<< mass[i][j]*fact << " in " << pos+i <<","<< pos+j <<endl;
            }
    }
};


#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void UniformMass<defaulttype::Affine3dTypes, defaulttype::Affine3dMass>::draw( const core::visual::VisualParams* vparams );
template <> SOFA_Flexible_API
double UniformMass<defaulttype::Affine3dTypes, defaulttype::Affine3dMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_Flexible_API
void UniformMass<defaulttype::Affine3fTypes, defaulttype::Affine3fMass>::draw( const core::visual::VisualParams* vparams );
template <> SOFA_Flexible_API
double UniformMass<defaulttype::Affine3fTypes, defaulttype::Affine3fMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif



} // namespace mass




} // namespace component



namespace core
{

namespace behavior
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_FLEXIBLECOMPATIBLITY_CPP)
extern template class SOFA_Flexible_API ForceField<defaulttype::Affine3Types>;
#endif

} // namespace behavior

} // namespace core


} // namespace sofa








#endif // FLEXIBLE_AffineComponents_H
