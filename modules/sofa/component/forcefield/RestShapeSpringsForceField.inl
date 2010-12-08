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
    , recompute_indices(initData(&recompute_indices,true, "recompute_indices", "Recompute indices (should be false for BBOX)"))
    , restMState(NULL)
{

}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::init()
{
    core::behavior::ForceField<DataTypes>::init();

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

//         for (unsigned int i = 0; i < points.getValue().size(); i++)
//         {
//             indices.push_back(i);
//         }
//
//         external_points.setValue(indices);
    }
    else
    {
        useRestMState = true;

        std::cout << "RestShapeSpringsForceField : Mechanical state named " << restMState->getName()
                << " found for RestShapeSpringFF named " << this->getName() << std::endl;

//         if (external_points.getValue().empty())
//         {
//             serr << "RestShapeSpringsForceField : external_points undefined, default case: external_points assigned " << sendl;
//
//             int pointSize = (int)points.getValue().size();
//             int restMstateSize = (int)restMState->getSize();
//
//             if (pointSize > restMstateSize)
//                 serr<<"ERROR in  RestShapeSpringsForceField<Rigid3fTypes>::init() : extenal_points must be defined !!" <<sendl;
//
//             for (unsigned int i = 0; i < points.getValue().size(); i++)
//             {
//                 indices.push_back(i);
//             }
//
//             external_points.setValue(indices);
//         }
    }


    for (unsigned int i = 0; i < points.getValue().size(); i++) this->indices.push_back(points.getValue()[i]);
    for (unsigned int i = 0; i < external_points.getValue().size(); i++) this->ext_indices.push_back(external_points.getValue()[i]);

    this->k = stiffness.getValue();
//     pp_0 = this->mstate->getX0();
    if (useRestMState) pp_0 = restMState->getX();
    else pp_0 = this->mstate->getX0();

    if (pp_0==NULL)
    {
        std::cerr << "Index not found in " << this->getName() << std::endl;
        indices.clear();
    }

    if (indices.size()==0)
    {
        std::cout << "in RestShapeSpringsForceField no point are defined, default case: points = all points " << std::endl;

        for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
        {
            indices.push_back(i);
        }
    }
    if (ext_indices.size()==0)
    {
        std::cout << "in RestShapeSpringsForceField no external_points are defined, default case: points = all points " << std::endl;

        if (useRestMState)
        {
            for (unsigned int i = 0; i < (unsigned)restMState->getSize(); i++)
            {
                ext_indices.push_back(i);
            }
        }
        else
        {
            for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
            {
                ext_indices.push_back(i);
            }
        }
    }

    if (this->indices.size()>this->ext_indices.size())
    {
        std::cerr << "Error : the dimention of the source and the targeted points are different " << std::endl;
        indices.clear();
    }

}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addForce(DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */, const core::MechanicalParams* /* mparams */)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;

    f1.resize(p1.size());

    if (recompute_indices.getValue())
    {
        indices.clear();
        ext_indices.clear();
        for (unsigned int i = 0; i < points.getValue().size(); i++) this->indices.push_back(points.getValue()[i]);
        for (unsigned int i = 0; i < external_points.getValue().size(); i++) this->ext_indices.push_back(external_points.getValue()[i]);

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

            Deriv dx = p1[index] - (*pp_0)[ext_index];
            Springs_dir[i] = p1[index] - (*pp_0)[ext_index];
            Springs_dir[i].normalize();
            f1[index] -=  dx * k[0] ;

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

            Deriv dx = p1[index] - (*pp_0)[ext_index];
            Springs_dir[i] = p1[index] - (*pp_0)[ext_index];
            Springs_dir[i].normalize();
            f1[index] -=  dx * k[index] ;

            //	if (dx.norm()>0.00000001)
            //		std::cout<<"force on point "<<index<<std::endl;

            //	Deriv dx = p[i] - p_0[i];
            //	f[ indices[i] ] -=  dx * k[i] ;
        }
    }
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addDForce(DataVecDeriv& df, const DataVecDeriv& dx, const core::MechanicalParams* mparams)
{
    //  remove to be able to build in parallel
    // 	const VecIndex& indices = points.getValue();
    // 	const VecReal& k = stiffness.getValue();

    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;
    double kFactor = mparams->kFactor();

    if (k.size()!= indices.size() )
    {
        sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;

        for (unsigned int i=0; i<indices.size(); i++)
        {
            df1[indices[i]] -=  dx1[indices[i]] * k[0] * kFactor;
        }
    }
    else
    {
        for (unsigned int i=0; i<indices.size(); i++)
        {
            df1[indices[i]] -=  dx1[indices[i]] * k[indices[i]] * kFactor ;
        }
    }
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, const core::MechanicalParams* mparams )
{
    //      remove to be able to build in parallel
    // 	const VecIndex& indices = points.getValue();
    // 	const VecReal& k = stiffness.getValue();

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    double kFact = mparams->kFactor();

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

#endif // SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL



