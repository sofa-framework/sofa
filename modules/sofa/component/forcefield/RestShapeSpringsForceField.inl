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
#ifndef SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL

#include <sofa/core/behavior/ForceField.inl>
#include "RestShapeSpringsForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>



namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
RestShapeSpringsForceField<DataTypes>::RestShapeSpringsForceField()
    : points(initData(&points, "points", "points controlled by the rest shape springs"))
    , stiffness(initData(&stiffness, "stiffness", "stiffness values between the actual position and the rest shape position"))
    , angularStiffness(initData(&angularStiffness, "angularStiffness", "angularStiffness assigned when controlling the rotation of the points"))
    , external_rest_shape(initData(&external_rest_shape, "external_rest_shape", "rest_shape can be defined by the position of an external Mechanical State"))
    , external_points(initData(&external_points, "external_points", "points from the external Mechancial State that define the rest shape springs"))
{}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::init()
{

    core::behavior::ForceField<DataTypes>::init();

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



    const std::string path = external_rest_shape.getValue();

    if (path.size()>0)
    {
        this->getContext()->get(restMState ,path  );
        //std::cout<< "for RestShapeSpringFF named "<<this->getName()<<", path = "<<path<<std::endl;
    }
    else
        restMState = NULL;



    VecIndex indices;
    if(restMState == NULL)
    {
        if(!external_rest_shape.getValue().empty())
            std::cout<<"do not found any Mechanical state named "<<external_rest_shape.getValue()<<std::endl;
        useRestMState = false;

        for (unsigned int i=0; i<points.getValue().size(); i++)
        {
            indices.push_back(i);
        }
        external_points.setValue(indices);

    }
    else
    {
        std::cout<<"Mechanical state named "<<restMState->getName()<< " found for RestShapeSpringFF named "<<this->getName()<<std::endl;
        useRestMState = true;
        if (external_points.getValue().size()==0)
        {

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


}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& p, const VecDeriv& )
{
    /*
    VecCoord &p_0;

    else
    	p_0 = *this->mstate->getX0();
    */


    VecCoord& p_0 = *this->mstate->getX0();
    //std::cout<<"p_0 in addForce"<<p_0<<std::endl;
    if (useRestMState)
        p_0 = *restMState->getX();

    /*
    std::cout<<"p_0 in addForce"<<p_0<<std::endl;
    //std::cout<<"addForce call in RestShapeSpringsForceField"<<std::endl;
    */

    f.resize(p.size());


    const VecIndex& indices = points.getValue();
    const VecIndex& ext_indices=external_points.getValue();
    const VecReal& k = stiffness.getValue();

    Springs_dir.resize(indices.size() );
    if ( k.size()!= indices.size() )
    {
        //sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;

        for (unsigned int i=0; i<indices.size(); i++)
        {
            const unsigned int index = indices[i];
            const unsigned int ext_index = ext_indices[i];

            Deriv dx = p[index] - p_0[ext_index];
            Springs_dir[i] = p[index] - p_0[ext_index];
            Springs_dir[i].normalize();
            f[index] -=  dx * k[0] ;

            //	if (dx.norm()>0.00000001)
            //		std::cout<<"force on point "<<index<<std::endl;

            //	Deriv dx = p[i] - p_0[i];
            //	f[ indices[i] ] -=  dx * k[0] ;
        }
    }
    else
    {
        for (unsigned int i=0; i<indices.size(); i++)
        {
            const unsigned int index = indices[i];
            const unsigned int ext_index = ext_indices[i];

            Deriv dx = p[index] - p_0[ext_index];
            Springs_dir[i] = p[index] - p_0[ext_index];
            Springs_dir[i].normalize();
            f[index] -=  dx * k[index] ;

            //	if (dx.norm()>0.00000001)
            //		std::cout<<"force on point "<<index<<std::endl;

            //	Deriv dx = p[i] - p_0[i];
            //	f[ indices[i] ] -=  dx * k[i] ;
        }
    }



}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addDForce(VecDeriv& df, const VecDeriv &dx, double kFactor, double )
{
    const VecIndex& indices = points.getValue();
    const VecReal& k = stiffness.getValue();

    if (k.size()!= indices.size() )
    {
        sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;

        for (unsigned int i=0; i<indices.size(); i++)
        {
            df[indices[i]] -=  dx[indices[i]] * k[0] * kFactor;
        }
    }
    else
    {
        for (unsigned int i=0; i<indices.size(); i++)
        {
            //	df[ indices[i] ] -=  dx[indices[i]] * k[i] * kFactor ;
            df[indices[i]] -=  dx[indices[i]] * k[indices[i]] * kFactor ;
        }
    }
    //serr<<"addDForce: dx = "<<dx<<"  - df = "<<df<<sendl;

}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * mat, double kFact, unsigned int &offset)
{
    const VecIndex& indices = points.getValue();
    const VecReal& k = stiffness.getValue();
    const int N = Coord::static_size;

    unsigned int curIndex = 0;


    if (k.size()!= indices.size() )
    {
        for (unsigned int index = 0; index < indices.size(); index++)
        {
            curIndex = indices[index];

            for(int i = 0; i < N; i++)
            {

                //	for (unsigned int j = 0; j < N; j++)
                //	{
                //		mat->add(offset + N * curIndex + i, offset + N * curIndex + j, kFact * k[0]);
                //	}

                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, - kFact * k[0]);
            }
        }
    }
    else
    {
        for (unsigned int index = 0; index < indices.size(); index++)
        {
            curIndex = indices[index];

            for(int i = 0; i < N; i++)
            {

                //	for (unsigned int j = 0; j < N; j++)
                //	{
                //		mat->add(offset + N * curIndex + i, offset + N * curIndex + j, kFact * k[curIndex]);
                //	}

                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k[curIndex]);
            }
        }
    }
}


//template <class DataTypes>
//double RestShapeSpringsForceField<DataTypes>::getPotentialEnergy(const VecCoord& x)
//{
//	const VecIndex& indices = points.getValue();
//	const VecDeriv& f = forces.getValue();
//	double e=0;
//	unsigned int i = 0;
//	for (; i<f.size(); i++)
//	{
//		e -= f[i]*x[indices[i]];
//	}
//	for (; i<indices.size(); i++)
//	{
//		e -= f[f.size()-1]*x[indices[i]];
//	}
//	return e;
//}


//template <class DataTypes>
//void RestShapeSpringsForceField<DataTypes>::setForce( unsigned i, const Deriv& force )
//{
//	VecIndex& indices = *points.beginEdit();
//	VecDeriv& f = *forces.beginEdit();
//	indices.push_back(i);
//	f.push_back( force );
//	points.endEdit();
//	forces.endEdit();
//}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::draw()
{
    /*
    if (!getContext()->getShowForceFields())
    	return;  /// \todo put this in the parent class


    const VecIndex& indices = points.getValue();
    const VecDeriv& f = forces.getValue();
    const VecCoord& x = *this->mstate->getX();
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor3f(0,1,0);
    for (unsigned int i=0; i<indices.size(); i++)
    {
    	Real xx,xy,xz,fx,fy,fz;
    	DataTypes::get(xx,xy,xz,x[indices[i]]);
    	DataTypes::get(fx,fy,fz,f[(i<f.size()) ? i : f.size()-1]);
    	glVertex3f( (GLfloat)xx, (GLfloat)xy, (GLfloat)xz );
    	glVertex3f( (GLfloat)(xx+fx), (GLfloat)(xy+fy), (GLfloat)(xz+fz) );
    }
    glEnd();
    */
}



template <class DataTypes>
bool RestShapeSpringsForceField<DataTypes>::addBBox(double*, double* )
{
    return false;
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif



