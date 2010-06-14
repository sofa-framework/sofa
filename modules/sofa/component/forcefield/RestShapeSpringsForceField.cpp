/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
template <>
void RestShapeSpringsForceField<Rigid3dTypes>::init()
{

    core::behavior::ForceField<Rigid3dTypes>::init();

    if (points.getValue().size()==0)
    {
        VecIndex indices; // = points.getValue();
        std::cout<<"in RestShapeSpringsForceField no point is defined, default case: points = all points "<<std::endl;

        for (unsigned int i=0; i<(unsigned)this->mstate->getSize(); i++)
        {
            indices.push_back(i);
        }
        points.setValue(indices);

    }
    if(stiffness.getValue().size() == 0)
    {
        VecReal stiffs;
        stiffs.push_back(100.0);
        std::cout<<"in RestShapeSpringsForceField no stiffness is defined, assuming equal stiffness on each node, k = 100.0 "<<std::endl;
        stiffness.setValue(stiffs);
    }

    if(stiffness.getValue().size() != points.getValue().size())
    {
        // The stiffness[0] is assigned to all points
        const VecIndex& indices = points.getValue();
        VecReal stiffs;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            stiffs.push_back(stiffness.getValue()[0]);
        }
        stiffness.setValue(stiffs);
    }

    if(angularStiffness.getValue().size() == 0)
    {
        VecReal Astiffs;
        Astiffs.push_back(10.0);
        std::cout<<"in RestShapeSpringsForceField no Angular stiffness is defined while using RigidTypes, assuming equal angular stiffness on each node, k = 10.0 "<<std::endl;
        angularStiffness.setValue(Astiffs);
    }

    if(angularStiffness.getValue().size() != points.getValue().size())
    {
        // The stiffness[0] is assigned to all points
        const VecIndex& indices = points.getValue();
        VecReal Astiffs;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Astiffs.push_back(angularStiffness.getValue()[0]);
        }
        angularStiffness.setValue(Astiffs);
    }






    const std::string path = external_rest_shape.getValue();
    if (path.size()>0)
    {
        this->getContext()->get(restMState ,path  );
    }
    else
        restMState = NULL;
    //this->getContext()->get(restMState ,path  );

    if(restMState == NULL)
    {
        if(!external_rest_shape.getValue().empty())
            std::cout<<"do not found any Mechanical state named "<<external_rest_shape.getValue()<<std::endl;
        useRestMState = false;
        external_points.setValue(points.getValue());
    }
    else
    {
        std::cout<<"Mechanical state named "<<restMState->getName()<< " found"<<std::endl;
        useRestMState = true;
        if (external_points.getValue().size()==0)
        {
            VecIndex indices; // = points.getValue();
            serr<<"in RestShapeSpringsForceField external_points undefined, external_points assigned automatically "<<sendl;

            int pointSize = (int) points.getValue().size();
            int restMstateSize = (int)  restMState->getSize();

            if (pointSize > restMstateSize)
                serr<<"ERROR in  RestShapeSpringsForceField<Rigid3fTypes>::init() : extenal_points must be defined !!" <<sendl;


            for (unsigned int i=0; i<points.getValue().size(); i++)
            {
                indices.push_back(i);
            }
            external_points.setValue(indices);
        }


    }



};


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addForce(VecDeriv& f, const VecCoord& p, const VecDeriv& )
{


    //const VecCoord& p_0 = *this->mstate->getX0();
    VecCoord p_0;
    if (useRestMState)
        p_0 = *restMState->getX();
    else
        p_0 = *this->mstate->getX0();

    //std::cout<<" AddForce : p_0 = "<<p_0<<" p ="<< p <<std::endl;

    //std::cout<<"addForce call in RestShapeSpringsForceField"<<std::endl;

    f.resize(p.size());

    const VecIndex& indices = points.getValue();
    const VecIndex& ext_indices = external_points.getValue();
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    for (unsigned int i=0; i<indices.size(); i++)
    {
        const unsigned int index = indices[i];
        const unsigned int ext_index = ext_indices[i];

        // translation
        Vec3f dx = p[index].getCenter() - p_0[ext_index].getCenter();
        f[index].getVCenter() -=  dx * k[i] ;

        // rotation
        Quat dq = p[index].getOrientation() * p_0[ext_index].getOrientation().inverse();
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

        //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<std::endl;
        f[index].getVOrientation() -= dir * angle * k_a[i] ;
    }

}


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addDForce(VecDeriv& df, const VecDeriv &dx, double kFactor, double )
{
    const VecIndex& indices = points.getValue();
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    //std::cout<<" kFactor :"<<kFactor<<std::endl;



    for (unsigned int i=0; i<indices.size(); i++)
    {
        df[indices[i]].getVCenter()		 -=  dx[indices[i]].getVCenter()	  * k[i]   * kFactor ;
        df[indices[i]].getVOrientation() -=  dx[indices[i]].getVOrientation() * k_a[i] * kFactor ;
    }

    //serr<<"addDForce : dx="<<dx<<"  - df="<<df<<sendl;

}


template<>
void RestShapeSpringsForceField<Rigid3dTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * mat, double kFact, unsigned int &offset)
{
    const VecIndex& indices = points.getValue();
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    const int N = 6;
    //std::cout<<"addKToMatrix<Rigid3dTypes> : N = "<<N<<"   - kFactor :"<<kFact<<std::endl;

    unsigned int curIndex = 0;


    for (unsigned int index = 0; index < indices.size(); index++)
    {
        curIndex = indices[index];

        // translation
        for(int i = 0; i < 3; i++)
        {
            mat->add(offset + N * curIndex + i, offset + N * curIndex + i,- kFact * k[index]);
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


#endif

#ifndef SOFA_DOUBLE
template <>
void RestShapeSpringsForceField<Rigid3fTypes>::init()
{

    core::behavior::ForceField<Rigid3fTypes>::init();

    if (points.getValue().size()==0)
    {
        VecIndex indices; // = points.getValue();
        std::cout<<"in RestShapeSpringsForceField no point is defined, default case: points = all points "<<std::endl;

        for (unsigned int i=0; i<(unsigned)this->mstate->getSize(); i++)
        {
            indices.push_back(i);
        }
        points.setValue(indices);

    }
    if(stiffness.getValue().size() == 0)
    {
        VecReal stiffs;
        stiffs.push_back(100.0f);
        std::cout<<"in RestShapeSpringsForceField no stiffness is defined, assuming equal stiffness on each node, k = 100.0 "<<std::endl;
        stiffness.setValue(stiffs);
    }

    if(stiffness.getValue().size() != points.getValue().size())
    {
        // The stiffness[0] is assigned to all points
        const VecIndex& indices = points.getValue();
        VecReal stiffs;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            stiffs.push_back(stiffness.getValue()[0]);
        }
        stiffness.setValue(stiffs);
    }

    if(angularStiffness.getValue().size() == 0)
    {
        VecReal Astiffs;
        Astiffs.push_back(10.0f);
        std::cout<<"in RestShapeSpringsForceField no Angular stiffness is defined while using RigidTypes, assuming equal angular stiffness on each node, k = 10.0 "<<std::endl;
        angularStiffness.setValue(Astiffs);
    }

    if(angularStiffness.getValue().size() != points.getValue().size())
    {
        // The stiffness[0] is assigned to all points
        const VecIndex& indices = points.getValue();
        VecReal Astiffs;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Astiffs.push_back(angularStiffness.getValue()[0]);
        }
        angularStiffness.setValue(Astiffs);
    }

    const std::string path = external_rest_shape.getValue();
    this->getContext()->get(restMState ,path  );

    if(restMState == NULL)
    {
        std::cout<<"do not found any Mechanical state named "<<external_rest_shape.getValue()<<std::endl;
        useRestMState = false;
    }
    else
    {
        std::cout<<"Mechanical state named "<<restMState->getName()<< " found"<<std::endl;
        useRestMState = true;
        if (external_points.getValue().size()==0)
        {
            VecIndex indices; // = points.getValue();
            serr<<"in RestShapeSpringsForceField external_points undefined, default case: external_points assigned "<<sendl;


            int pointSize = (int)points.getValue().size();
            int restMstateSize = (int)  restMState->getSize();

            if (  pointSize>restMstateSize)
                serr<<"ERROR in  RestShapeSpringsForceField<Rigid3fTypes>::init() : extenal_points must be defined !!" <<sendl;

            for (unsigned int i=0; i<points.getValue().size(); i++)
            {
                indices.push_back(i);
            }
            external_points.setValue(indices);
        }


    }

};


template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addForce(VecDeriv& f, const VecCoord& p, const VecDeriv& )
{
    VecCoord p_0;
    if (useRestMState)
        p_0 = *restMState->getX();
    else
        p_0 = *this->mstate->getX0();








    //std::cout<<"addForce call in RestShapeSpringsForceField"<<std::endl;

    f.resize(p.size());

    const VecIndex& indices = points.getValue();
    const VecIndex& ext_indices=external_points.getValue();
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    for (unsigned int i=0; i<indices.size(); i++)
    {
        const unsigned int index = indices[i];
        const unsigned int ext_index = ext_indices[i];

        // translation
        Vec3f dx = p[index].getCenter() - p_0[ext_index].getCenter();
        f[index].getVCenter() -=  dx * k[i] ;

        // rotation
        Quat dq = p[index].getOrientation() * p_0[ext_index].getOrientation().inverse();
        Vec3d dir;
        double angle=0;
        dq.normalize();
        if (dq[3] < 0.999999999999999)
            dq.quatToAxis(dir, angle);
        dq.quatToAxis(dir, angle);

        //std::cout<<"dq : "<<dq <<"  dir :"<<dir<<"  angle :"<<angle<<std::endl;
        f[index].getVOrientation() -= dir * angle * k_a[i] ;
    }

}

template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addDForce(VecDeriv& df, const VecDeriv &dx, double kFactor, double )
{
    const VecIndex& indices = points.getValue();
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();

    for (unsigned int i=0; i<indices.size(); i++)
    {
        df[indices[i]].getVCenter()		 -=  dx[indices[i]].getVCenter()	  * k[i]   * kFactor ;
        df[indices[i]].getVOrientation() -=  dx[indices[i]].getVOrientation() * k_a[i] * kFactor ;
    }

}



template<>
void RestShapeSpringsForceField<Rigid3fTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * mat, double kFact, unsigned int &offset)
{
    const VecIndex& indices = points.getValue();
    const VecReal& k = stiffness.getValue();
    const VecReal& k_a = angularStiffness.getValue();
    const int N = 6;
    //std::cout<<"addKToMatrix : N = "<<N<<std::endl;

    unsigned int curIndex = 0;


    for (unsigned int index = 0; index < indices.size(); index++)
    {
        curIndex = indices[index];

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

#endif

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
void RestShapeSpringsForceField<Vec3dTypes>::draw()
{

    if (!this->getContext()->getShowForceFields())
        return;  /// \todo put this in the parent class

    VecCoord& p_0 = *this->mstate->getX0();
    if (useRestMState)
        p_0 = *restMState->getX();

    //std::cout<<"p_0 in draw : "<<p_0<<std::endl;

    VecCoord& p = *this->mstate->getX();


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
        glVertex3f( (GLfloat)p_0[ext_index][0], (GLfloat)p_0[ext_index][1], (GLfloat)p_0[ext_index][2] );

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
template class RestShapeSpringsForceField<Vec3dTypes>;
//template class RestShapeSpringsForceField<Vec2dTypes>;
template class RestShapeSpringsForceField<Vec1dTypes>;
//template class RestShapeSpringsForceField<Vec6dTypes>;
template class RestShapeSpringsForceField<Rigid3dTypes>;
//template class RestShapeSpringsForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class RestShapeSpringsForceField<Vec3fTypes>;
//template class RestShapeSpringsForceField<Vec2fTypes>;
template class RestShapeSpringsForceField<Vec1fTypes>;
//template class RestShapeSpringsForceField<Vec6fTypes>;
template class RestShapeSpringsForceField<Rigid3fTypes>;
//template class RestShapeSpringsForceField<Rigid2fTypes>;
#endif



} // namespace forcefield

} // namespace component

} // namespace sofa
