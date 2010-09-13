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
    , recomput_indices(initData(&recomput_indices,true, "recomput_indices", "Recompute indices (should be false for BBOX)"))
    , restMState(NULL)
{

}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::init()
{
    core::behavior::ForceField<DataTypes>::init();

    if (points.getValue().empty())
    {
        VecIndex indices;
        std::cout << "in RestShapeSpringsForceField no point is defined, default case: points = all points " << std::endl;

        for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
        {
            indices.push_back(i);
        }

        points.setValue(indices);
    }

    if (stiffness.getValue().empty())
    {
        std::cout << "RestShapeSpringsForceField : No stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << std::endl;

        VecReal stiffs;
        stiffs.push_back(100.0);
        stiffness.setValue(stiffs);
    }

    const std::string path = external_rest_shape.getValue();

    restMState = NULL;

    if (path.size() > 0)
    {
        this->getContext()->get(restMState ,path);
    }

    VecIndex indices;

    if (!restMState)
    {
        useRestMState = false;

        if (path.size() > 0)
        {
            std::cout << "RestShapeSpringsForceField : " << external_rest_shape.getValue() << "not found\n";
        }

        for (unsigned int i = 0; i < points.getValue().size(); i++)
        {
            indices.push_back(i);
        }

        external_points.setValue(indices);
    }
    else
    {
        useRestMState = true;

        std::cout << "RestShapeSpringsForceField : Mechanical state named " << restMState->getName()
                << " found for RestShapeSpringFF named " << this->getName() << std::endl;

        if (external_points.getValue().empty())
        {
            serr << "RestShapeSpringsForceField : external_points undefined, default case: external_points assigned " << sendl;

            int pointSize = (int)points.getValue().size();
            int restMstateSize = (int)restMState->getSize();

            if (pointSize > restMstateSize)
                serr<<"ERROR in  RestShapeSpringsForceField<Rigid3fTypes>::init() : extenal_points must be defined !!" <<sendl;

            for (unsigned int i = 0; i < points.getValue().size(); i++)
            {
                indices.push_back(i);
            }

            external_points.setValue(indices);
        }
    }

    this->indices = points.getValue();
    this->ext_indices = external_points.getValue();
    this->k = stiffness.getValue();
    pp_0 = this->mstate->getX0();
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addForce(VecDeriv& f, const VecCoord& p, const VecDeriv& )
{
    if (recomput_indices.getValue()) pp_0 = this->mstate->getX0();

    VecCoord& p_0 = *pp_0;

    if (useRestMState)
        p_0 = *restMState->getX();

    f.resize(p.size());

    if (recomput_indices.getValue())
    {
        indices = points.getValue();
        ext_indices = external_points.getValue();
        stiffness.getValue();
    }

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
//      remove to be able to build in parallel
// 	const VecIndex& indices = points.getValue();
// 	const VecReal& k = stiffness.getValue();

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
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * mat, double kFact, unsigned int &offset)
{
//      remove to be able to build in parallel
// 	const VecIndex& indices = points.getValue();
// 	const VecReal& k = stiffness.getValue();

    const int N = Coord::total_size;

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

                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k[0]);
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


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::draw()
{

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



