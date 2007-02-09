#include <sofa/component/constraint/OscillatorConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/helper/gl/Axis.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;

//template <>
//void OscillatorConstraint<RigidTypes>::draw()
//{
///*	if (!getContext()->getShowBehaviorModels()) return;
//	VecCoord& x = *mstate->getX();
//	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
//	{
//		int i = *it;
//		Quat orient = x[i].getOrientation();
//		RigidTypes::Vec3& center = x[i].getCenter();
//		orient[3] = -orient[3];
//
//		static GL::Axis *axis = new GL::Axis(center, orient, 5);
//
//		axis->update(center, orient);
//		axis->draw();
//	}*/
//	if (!getContext()->getShowBehaviorModels()) return;
//	VecCoord& x = *mstate->getX();
//	glDisable (GL_LIGHTING);
//	glPointSize(10);
//	glColor4f (1,0.5,0.5,1);
//	glBegin (GL_POINTS);
//	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
//	{
//	   helper::gl::glVertexT(x[0].getCenter());
//	}
//	glEnd();
//}
//
//template <>
//      void OscillatorConstraint<RigidTypes>::projectResponse(VecDeriv& res)
//{
//   res[0] = Deriv();
//}


SOFA_DECL_CLASS(OscillatorConstraint)

template class OscillatorConstraint<Vec3dTypes>;
template class OscillatorConstraint<Vec3fTypes>;


int OscillatorConstraintClass = core::RegisterObject("TODO")
        .add< OscillatorConstraint<Vec3dTypes> >()
        .add< OscillatorConstraint<Vec3fTypes> >()
        .add< OscillatorConstraint<RigidTypes> >()
        ;

} // namespace constraint

} // namespace component

} // namespace sofa

