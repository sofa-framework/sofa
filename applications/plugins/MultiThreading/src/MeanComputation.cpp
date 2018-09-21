#include "MeanComputation.inl"

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

    namespace component
    {

        namespace engine
        {

            SOFA_DECL_CLASS(MeanComputation)

                int MeanComputationEngineClass = core::RegisterObject("Compute the mean of the input elements")
#ifdef SOFA_FLOAT
                .add< MeanComputationEngine<defaulttype::Vec3fTypes> >(true) // default template
#else
                .add< MeanComputation<defaulttype::Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
                .add< MeanComputation<defaulttype::Vec3fTypes> >()
#endif
#endif
#ifndef SOFA_FLOAT
                .add< MeanComputation<defaulttype::Vec1dTypes> >()
                .add< MeanComputation<defaulttype::Vec2dTypes> >()
                .add< MeanComputation<defaulttype::Rigid2dTypes> >()
                .add< MeanComputation<defaulttype::Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
                .add< MeanComputation<defaulttype::Vec1fTypes> >()
                .add< MeanComputation<defaulttype::Vec2fTypes> >()
                .add< MeanComputation<defaulttype::Rigid2fTypes> >()
                .add< MeanComputation<defaulttype::Rigid3fTypes> >()
#endif //SOFA_DOUBLE
                ;

#ifndef SOFA_FLOAT
            template class MeanComputation<defaulttype::Vec1dTypes>;
            template class MeanComputation<defaulttype::Vec2dTypes>;
            template class MeanComputation<defaulttype::Vec3dTypes>;
            template class MeanComputation<defaulttype::Rigid2dTypes>;
            template class MeanComputation<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
            template class MeanComputation<defaulttype::Vec1fTypes>;
            template class MeanComputation<defaulttype::Vec2fTypes>;
            template class MeanComputation<defaulttype::Vec3fTypes>;
            template class MeanComputation<defaulttype::Rigid2fTypes>;
            template class MeanComputation<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE


        } // namespace constraint

    } // namespace component

} // namespace sofa