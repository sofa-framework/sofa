#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/component/constraint/FixedPlaneConstraint.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint()
    : direction(0.0,0.0,1.0)
{
    selectVerticesFromPlanes=false;
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::~FixedPlaneConstraint()
{
}


template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherit::parse(arg);
    FixedPlaneConstraint<DataTypes>* obj = this;
    if (arg->getAttribute("indices"))
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
    }
    if (arg->getAttribute("direction"))
    {
        const char* str = arg->getAttribute("direction");
        const char* str2 = NULL;
        Real val[3];
        unsigned int i;
        for(i=0; i<3; i++)
        {
            val[i] = (Real)strtod(str,(char**)&str2);
            if (str2==str) break;
            str = str2;
        }
        Coord dir(val);
        obj->setDirection(dir);
    }
    if (arg->getAttribute("distance"))
    {
        const char* str = arg->getAttribute("distance");
        const char* str2 = NULL;
        Real val[2];
        unsigned int i;
        for(i=0; i<2; i++)
        {
            val[i] = (Real)strtod(str,(char**)&str2);
            if (str2==str) break;
            str = str2;
        }
        obj->setDminAndDmax(val[0],val[1]);
    }
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>*  FixedPlaneConstraint<DataTypes>::addConstraint(int index)
{
    this->indices.insert(index);
    return this;
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>*  FixedPlaneConstraint<DataTypes>::removeConstraint(int index)
{
    this->indices.erase(index);
    return this;
}

// -- Mass interface
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectResponse(VecDeriv& res)
{

    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        /// only constraint one projection of the displacement to be zero
        res[*it]-= direction*dot(res[*it],direction);
    }

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDirection(Coord dir)
{
    if (dir.norm2()>0)
    {
        direction=dir;
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::selectVerticesAlongPlane()
{
    VecCoord& x = *this->mstate->getX();
    unsigned int i;
    for(i=0; i<x.size(); ++i)
    {
        if (isPointInPlane(x[i]))
            addConstraint(i);
    }

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::init()
{
    if (selectVerticesFromPlanes)
        selectVerticesAlongPlane();

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
    const VecCoord& x = *this->mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,1.0,0.5,1);
    glBegin (GL_POINTS);
    for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    {
        helper::gl::glVertexT(x[*it]);
    }
    glEnd();
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif
