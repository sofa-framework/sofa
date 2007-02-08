#include <sofa/component/constraint/OscillatorConstraint.inl>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/helper/gl/Axis.h>
#include <sstream>

namespace sofa
{

namespace helper   // \todo Why this must be inside helper namespace
{

using namespace component::constraint;

/** Clear the container and fill it with values read from the string */
template<class C>
void readVector( C& container, const char* string )
{
    typedef typename C::value_type value_type;
    value_type v;
    container.clear();
    std::istringstream in(string);
    while( in >> v )
        container.push_back(v);
}

template<class DataTypes>
void create(OscillatorConstraint<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< OscillatorConstraint<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (const char* str = arg->getAttribute("indices"))
        {
            vector<unsigned> indices;
            readVector(indices,str);
            //const char* str = arg->getAttribute("indices");
            //const char* str2 = NULL;
            //for(;;)
            //{
            //	int v = (int)strtod(str,(char**)&str2);
            //	if (str2==str) break;
            //	str = str2;
            //	obj->addConstraint(v);
            //}
        }
    }
}
}

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

using helper::Creator;

Creator<simulation::tree::xml::ObjectFactory, OscillatorConstraint<Vec3dTypes> > OscillatorConstraint3dClass("OscillatorConstraint",true);
Creator<simulation::tree::xml::ObjectFactory, OscillatorConstraint<Vec3fTypes> > OscillatorConstraint3fClass("OscillatorConstraint",true);
Creator<simulation::tree::xml::ObjectFactory, OscillatorConstraint<RigidTypes> > OscillatorConstraintRigidClass("OscillatorConstraint",true);

} // namespace constraint

} // namespace component

} // namespace sofa

