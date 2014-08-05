

#include "../initFlexible.h"


#include <SofaBaseMechanics/MechanicalObject.h>


#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <SofaBaseMechanics/UniformMass.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/Mass.h>


#ifdef SOFA_HAVE_IMAGE
#include "../mass/ImageDensityMass.h"
#endif


#include "ComponentSpecializationsDefines.h"





namespace sofa
{


// ==========================================================================
// Mechanical Object

namespace component
{
namespace container
{

#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void MechanicalObject<defaulttype::TYPEABSTRACTNAME3dTypes>::draw(const core::visual::VisualParams* vparams);
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_Flexible_API
void MechanicalObject<defaulttype::TYPEABSTRACTNAME3fTypes>::draw(const core::visual::VisualParams* vparams);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_COMPILING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::TYPEABSTRACTNAME3dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::TYPEABSTRACTNAME3fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::TYPEABSTRACTNAME3fTypes>;
#endif
#endif


} // namespace container







// ==========================================================================
// Uniform Mass


namespace mass
{

template<int N, typename Real>
class AddMToMatrixFunctor< typename defaulttype::StdTYPEABSTRACTNAMETypes<N,Real>::Deriv, defaulttype::DeformableFrameMass<N,defaulttype::StdTYPEABSTRACTNAMETypes<N,Real>::deriv_total_size,Real> >
{
public:
    void operator()(defaulttype::BaseMatrix * mat, const defaulttype::DeformableFrameMass<N,defaulttype::StdTYPEABSTRACTNAMETypes<N,Real>::deriv_total_size,Real>& mass, int pos, double fact)
    {
        typedef defaulttype::DeformableFrameMass<N,defaulttype::StdTYPEABSTRACTNAMETypes<N,Real>::deriv_total_size,Real> TYPEABSTRACTNAMEMass;
        for( unsigned i=0; i<TYPEABSTRACTNAMEMass::VSize; ++i )
            for( unsigned j=0; j<TYPEABSTRACTNAMEMass::VSize; ++j )
            {
                mat->add(pos+i, pos+j, mass[i][j]*fact);
//            cerr<<"AddMToMatrixFunctor< defaulttype::Vec<N,Real>, defaulttype::Mat<N,N,Real> >::operator(), add "<< mass[i][j]*fact << " in " << pos+i <<","<< pos+j <<endl;
            }
    }
};


#ifndef SOFA_FLOAT
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::draw( const core::visual::VisualParams* vparams );
template <> SOFA_Flexible_API
double UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes, defaulttype::TYPEABSTRACTNAME3dMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_Flexible_API
void UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::draw( const core::visual::VisualParams* vparams );
template <> SOFA_Flexible_API
double UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes, defaulttype::TYPEABSTRACTNAME3fMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif




#ifdef SOFA_HAVE_IMAGE

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_COMPILING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API ImageDensityMass<defaulttype::TYPEABSTRACTNAME3dTypes,core::behavior::ShapeFunctiond,defaulttype::TYPEABSTRACTNAME3dMass>;
extern template class SOFA_Flexible_API UniformMass<defaulttype::TYPEABSTRACTNAME3dTypes,defaulttype::TYPEABSTRACTNAME3dMass>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API ImageDensityMass<defaulttype::TYPEABSTRACTNAME3fTypes,core::behavior::ShapeFunctionf,defaulttype::TYPEABSTRACTNAME3fMass>;
extern template class SOFA_Flexible_API UniformMass<defaulttype::TYPEABSTRACTNAME3fTypes,defaulttype::TYPEABSTRACTNAME3fMass>;
#endif
#endif

#endif



} // namespace mass



} // namespace component



namespace core
{

namespace behavior
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_COMPILING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API ForceField<defaulttype::TYPEABSTRACTNAME3dTypes>;
extern template class SOFA_Flexible_API Mass<defaulttype::TYPEABSTRACTNAME3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API ForceField<defaulttype::TYPEABSTRACTNAME3fTypes>;
extern template class SOFA_Flexible_API Mass<defaulttype::TYPEABSTRACTNAME3fTypes>;
#endif
#endif


} // namespace behavior

} // namespace core




}// namespace sofa


#include "ComponentSpecializationsUndef.h"

