#ifndef FLEXIBLE_QuadraticComponents_H
#define FLEXIBLE_QuadraticComponents_H

#include "../initFlexible.h"

#include "QuadraticTypes.h"

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
void MechanicalObject<defaulttype::Quadratic3Types>::draw(const core::visual::VisualParams* vparams);

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_QuadraticComponents_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Quadratic3dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Quadratic3fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Quadratic3fTypes>;
#endif
#endif


} // namespace container







// ==========================================================================
// Uniform Mass


namespace mass
{

template<int N, typename Real>
class AddMToMatrixFunctor< typename defaulttype::StdQuadraticTypes<N,Real>::Deriv, defaulttype::QuadraticMass<N,Real> >
{
public:
    void operator()(defaulttype::BaseMatrix * mat, const defaulttype::QuadraticMass<N,Real>& mass, int pos, double fact)
    {
        typedef defaulttype::QuadraticMass<N,Real> QuadraticMass;
        for( unsigned i=0; i<QuadraticMass::VSize; ++i )
            for( unsigned j=0; j<QuadraticMass::VSize; ++j )
            {
                mat->add(pos+i, pos+j, mass[i][j]*fact);
//            cerr<<"AddMToMatrixFunctor< defaulttype::Vec<N,Real>, defaulttype::Mat<N,N,Real> >::operator(), add "<< mass[i][j]*fact << " in " << pos+i <<","<< pos+j <<endl;
            }
    }
};


#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void UniformMass<defaulttype::Quadratic3dTypes, defaulttype::Quadratic3dMass>::draw( const core::visual::VisualParams* vparams );
template <> SOFA_Flexible_API
double UniformMass<defaulttype::Quadratic3dTypes, defaulttype::Quadratic3dMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_Flexible_API
void UniformMass<defaulttype::Quadratic3fTypes, defaulttype::Quadratic3fMass>::draw( const core::visual::VisualParams* vparams );
template <> SOFA_Flexible_API
double UniformMass<defaulttype::Quadratic3fTypes, defaulttype::Quadratic3fMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif

} // namespace mass

} // namespace component



namespace core
{

namespace behavior
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_AffineComponents_CPP)
extern template class SOFA_Flexible_API ForceField<defaulttype::Quadratic3Types>;
#endif

} // namespace behavior

} // namespace core


} // namespace sofa








#endif // FLEXIBLE_QuadraticComponents_H
