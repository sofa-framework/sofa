/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_CPP

#include <sofa/component/forcefield/RestShapeSpringsForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(RestShapeSpringsForceField)

///////////// SPECIALIZATION FOR RIGID TYPES //////////////


#ifndef SOFA_FLOAT

template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addSpringForce(Deriv &/*f*/, const Coord &/*x0*/, const Coord &/*x1*/, const Deriv &/*v0*/, const Deriv &/*v1*/, const Real &/*k*/, const Real &/*kd*/)
{

}

template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addSpringForce(Deriv &f, const Coord &x0, const Coord &x1, const Deriv &v0, const Deriv &v1, const Real &k, const Real &kd, const Real &k_a, const Real &kd_a, CPos *pivot)
{
    // translation
    if (pivot == 0)
    {
        Vec3d dx = x1.getCenter() - x0.getCenter();
        Vec3d dv = v1.getVCenter() - v0.getVCenter();

        Vec3d u = dx;
        u.normalize();

        const Real forceIntensity = dx.norm() * k +  dot(u, dv) * kd;

        getVCenter(f) -=  dx * forceIntensity;
    }
    else
    {
        const CPos &piv = *pivot;

        CPos localPivot = x0.getOrientation().inverseRotate(piv - x0.getCenter());
        //std::cout << "localPivot = " << localPivot << std::endl;
        CPos rotatedPivot = x1.getOrientation().rotate(localPivot);
        CPos pivot2 = x1.getCenter() + rotatedPivot;
        CPos dx = pivot2 - piv;

        // @TODO Manage damping

        getVCenter(f) -= dx * k;
        //getVOrientation(f1[index]) -= cross(rotatedPivot, dx * k[i]);
    }

    // rotation
    Quatd dq = x1.getOrientation() * x0.getOrientation().inverse();
    Vec3d dv = v1.getVOrientation() - v0.getVOrientation();
    Vec3d dir;
    double angle=0;
    dq.normalize();

    if (dq[3] < 0)
    {
        //std::cout<<"WARNING inversion quaternion"<<std::endl;
        dq = dq * -1.0;
    }

    if (dq[3] < 0.999999999999999)
        dq.quatToAxis(dir, angle);

    //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<"  index : "<<index<<"  f1.size() : "<<f1.size()<<std::endl;
    //Vec3d m1 = getVOrientation(f1[index]) ;
    //std::cout<<"m1 = "<<m1<<std::endl;

    getVOrientation(f) -= (dir * angle * k_a + dv * kd_a);

    //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<std::endl;
}

template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p0 = *(useRestMState ? restMState->read(core::VecCoordId::position()) : this->mstate->read(core::VecCoordId::restPosition()));

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v1 = v;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v0 = *(useRestMState ? restMState->read(core::VecDerivId::velocity()) : this->mstate->read(core::VecDerivId::null()));

    //std::cout<<"addForce with p_0 ="<<p_0<<" getX0"<<(*this->mstate->getX0())<<std::endl;

    f1.resize(p1.size());
    //std::cout<<" size p1:"<<p1.size()<<std::endl;

    if (recompute_indices.getValue())
    {
        recomputeIndices();
    }

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    const VecReal& kd = m_damping.getValue();
    const VecReal& kd_a = m_angularDamping.getValue();

    for (unsigned int i = 0; i < m_indices.size(); i++)
    {
        const unsigned int index = m_indices[i];
        const unsigned int ext_index = m_ext_indices[i];

        if (i >= m_pivots.size())
        {
            addSpringForce(f1[index], p0[ext_index], p1[index], v0[ext_index], v1[index], k[i], kd[i], k_a[i], kd_a[i]);
        }
        else
        {
            addSpringForce(f1[index], p0[ext_index], p1[index], v0[ext_index], v1[index], k[i], kd[i], k_a[i], kd_a[i], &(m_pivots[i]));
        }
    }
}


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    double kFactor = mparams->kFactor();

    unsigned int curIndex = 0;

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
        curIndex = m_indices[i];
        getVCenter(df1[curIndex])	 -=  getVCenter(dx1[curIndex]) * k[i] * kFactor ;
        getVOrientation(df1[curIndex]) -=  getVOrientation(dx1[curIndex]) * k_a[i] * kFactor ;
    }
}


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addKToMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    const int N = 6;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    double kFact = mparams->kFactor();

    unsigned int curIndex = 0;

    for (unsigned int index = 0; index < m_indices.size(); index++)
    {
        curIndex = m_indices[index];

        // translation
        for(int i = 0; i < 3; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k[index]);
        }

        // rotation
        for(int i = 3; i < 6; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k_a[index]);
        }
    }


    /* debug
    std::cout<<"MAT obtained : size: ("<<mat->rowSize()<<" * "<<mat->colSize()<<")\n"<<std::endl;

    for (unsigned int col=0; col<mat->colSize(); col++)
    {
    	for (unsigned int row=0; row<mat->rowSize(); row++)
    	{
    			std::cout<<" "<<mat->element(row, col);
    	}

    	std::cout<<""<<std::endl;
    }
    */
}


#endif // SOFA_FLOAT

#ifndef SOFA_DOUBLE

template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addSpringForce(Deriv &/*f*/, const Coord &/*x0*/, const Coord &/*x1*/, const Deriv &/*v0*/, const Deriv &/*v1*/, const Real &/*k*/, const Real &/*kd*/)
{

}

template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p0 = *(useRestMState ? restMState->read(core::VecCoordId::position()) : this->mstate->read(core::VecCoordId::restPosition()));

    f1.resize(p1.size());

    if (recompute_indices.getValue())
    {
        recomputeIndices();
    }

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
        const unsigned int index = m_indices[i];
        const unsigned int ext_index = m_ext_indices[i];

        // translation
        Vec3f dx = p1[index].getCenter() - p0[ext_index].getCenter();
        getVCenter(f1[index]) -=  dx * k[i] ;

        // rotation
        Quatf dq = p1[index].getOrientation() * p0[ext_index].getOrientation().inverse();
        Vec3f dir;
        Real angle=0;
        dq.normalize();
        if (dq[3] < 0.999999999999999)
            dq.quatToAxis(dir, angle);
        dq.quatToAxis(dir, angle);

        //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<std::endl;
        getVOrientation(f1[index]) -= dir * angle * k_a[i] ;
    }
}


template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;
    double kFactor = mparams->kFactor();

    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    unsigned int curIndex = 0;

    for (unsigned int i=0; i<m_indices.size(); i++)
    {
//		curIndex = m_indices[index];
        curIndex = m_indices[i];  // Fix by FF, just a guess
        getVCenter(df1[curIndex])      -=  getVCenter(dx1[curIndex])      * k[i]   * kFactor ;
        getVOrientation(df1[curIndex]) -=  getVOrientation(dx1[curIndex]) * k_a[i] * kFactor ;
    }
}


template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addKToMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    const int N = 6;

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    double kFact = mparams->kFactor();

    unsigned int curIndex = 0;


    for (unsigned int index = 0; index < m_indices.size(); index++)
    {
        curIndex = m_indices[index];

        // translation
        for(int i = 0; i < 3; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, kFact * k[index]);
        }

        // rotation
        for(int i = 3; i < 6; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i, kFact * k_a[index]);
        }
    }
}

#endif // SOFA_DOUBLE

#ifndef SOFA_FLOAT

/*
template<>
void RestShapeSpringsForceField<Vec3dTypes>::addDForce(VecDeriv& df, const VecDeriv &dx, double kFactor, double )
{
const VecIndex& indices = points.getValue();
const VecReal& k = stiffness.getValue();

if (k.size()!= indices.size() )
{
    sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;

    for (unsigned int i=0; i<indices.size(); i++)
    {
            df[indices[i]] -=  Springs_dir[i]  * k[0] * kFactor * dot(dx[indices[i]], Springs_dir[i]);
    }
}
else
{
    for (unsigned int i=0; i<indices.size(); i++)
    {
    //	df[ indices[i] ] -=  dx[indices[i]] * k[i] * kFactor ;
            df[indices[i]] -=   Springs_dir[i]  * k[indices[i]] * kFactor * dot(dx[indices[i]] , Springs_dir[i]);
    }
}
}
*/


template<>
void RestShapeSpringsForceField<Vec3dTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;  /// \todo put this in the parent class

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p0 = *(useRestMState ? restMState->read(core::VecCoordId::position()) : this->mstate->read(core::VecCoordId::restPosition()));

    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p = this->mstate->read(core::VecCoordId::position());

    const VecIndex& indices = points.getValue();
    const VecIndex& ext_indices=external_points.getValue();


    for (unsigned int i=0; i<indices.size(); i++)
    {
        const unsigned int index = indices[i];
        const unsigned int ext_index = ext_indices[i];

        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        glColor3f(0,1,0);

        glVertex3f( (GLfloat)p[index][0], (GLfloat)p[index][1], (GLfloat)p[index][2] );
        glVertex3f( (GLfloat)p0[ext_index][0], (GLfloat)p0[ext_index][1], (GLfloat)p0[ext_index][2] );

        glEnd();
    }


}
#endif



int RestShapeSpringsForceFieldClass = core::RegisterObject("Simple elastic springs applied to given degrees of freedom between their current and rest shape position")
#ifndef SOFA_FLOAT
        .add< RestShapeSpringsForceField<Vec3dTypes> >()
//.add< RestShapeSpringsForceField<Vec2dTypes> >()
        .add< RestShapeSpringsForceField<Vec1dTypes> >()
//.add< RestShapeSpringsForceField<Vec6dTypes> >()
        .add< RestShapeSpringsForceField<Rigid3dTypes> >()
//.add< RestShapeSpringsForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< RestShapeSpringsForceField<Vec3fTypes> >()
//.add< RestShapeSpringsForceField<Vec2fTypes> >()
        .add< RestShapeSpringsForceField<Vec1fTypes> >()
//.add< RestShapeSpringsForceField<Vec6fTypes> >()
        .add< RestShapeSpringsForceField<Rigid3fTypes> >()
//.add< RestShapeSpringsForceField<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec3dTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec2dTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec1dTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec6dTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid3dTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec3fTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec2fTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec1fTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Vec6fTypes>;
template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid3fTypes>;
//template class SOFA_DEFORMABLE_API RestShapeSpringsForceField<Rigid2fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
