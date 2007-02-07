#include <sofa/component/constraint/FixedConstraint.inl>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace helper   // \todo Why this must be inside helper namespace
{
using namespace component::constraint;
template<class DataTypes>
void create(FixedConstraint<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< FixedConstraint<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
        /*		if (arg->getAttribute("indices"))
        		{
        			const char* str = arg->getAttribute("indices");
        			const char* str2 = NULL;
        			for(;;)
        			{
        				int v = (int)strtod(str,(char**)&str2);
        				if (str2==str) break;
        				str = str2;
        				obj->addConstraint(v);
        			}
        		}*/
    }
}
}

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;

template <>
void FixedConstraint<RigidTypes>::draw()
{
    /*	if (!getContext()->getShowBehaviorModels()) return;
    	VecCoord& x = *mstate->getX();
    	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    	{
    		int i = *it;
    		Quat orient = x[i].getOrientation();
    		RigidTypes::Vec3& center = x[i].getCenter();
    		orient[3] = -orient[3];

    		static GL::Axis *axis = new GL::Axis(center, orient, 5);

    		axis->update(center, orient);
    		axis->draw();
    	}*/
    const SetIndexArray & indices = f_indices.getValue().getArray();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        gl::glVertexT(x[0].getCenter());
    }
    glEnd();
}

template <>
void FixedConstraint<RigidTypes>::projectResponse(VecDeriv& res)
{
    res[0] = Deriv();
}


SOFA_DECL_CLASS(FixedConstraint)

template class FixedConstraint<Vec3dTypes>;
template class FixedConstraint<Vec3fTypes>;

using helper::Creator;

Creator<simulation::tree::xml::ObjectFactory, FixedConstraint<Vec3dTypes> > FixedConstraint3dClass("FixedConstraint",true);
Creator<simulation::tree::xml::ObjectFactory, FixedConstraint<Vec3fTypes> > FixedConstraint3fClass("FixedConstraint",true);
Creator<simulation::tree::xml::ObjectFactory, FixedConstraint<RigidTypes> > FixedConstraintRigidClass("FixedConstraint",true);

} // namespace constraint

} // namespace component

} // namespace sofa

