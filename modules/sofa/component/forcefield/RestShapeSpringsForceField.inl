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
#ifndef SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL

#include <sofa/core/behavior/ForceField.inl>
#include "RestShapeSpringsForceField.h"
#include <sofa/core/visual/VisualParams.h>
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
    , m_damping(initData(&m_damping, "damping", "damping on stiffness values between the actual position and the rest shape position"))
    , m_angularDamping(initData(&m_angularDamping, "angularDamping", "damping on angularStiffness" ))
    , pivotPoints(initData(&pivotPoints, "pivot_points", "global pivot points used when translations instead of the rigid mass centers"))
    , external_rest_shape(initData(&external_rest_shape, "external_rest_shape", "rest_shape can be defined by the position of an external Mechanical State"))
    , external_points(initData(&external_points, "external_points", "points from the external Mechancial State that define the rest shape springs"))
    , recompute_indices(initData(&recompute_indices, false, "recompute_indices", "Recompute indices (should be false for BBOX)"))
    , restMState(NULL)
//	, pp_0(NULL)
{

}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::bwdInit()
{
    core::behavior::ForceField<DataTypes>::init();

    if (stiffness.getValue().empty())
    {
        std::cout << "RestShapeSpringsForceField : No stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << std::endl;

        VecReal stiffs;
        stiffs.push_back(100.0);
        stiffness.setValue(stiffs);
    }

    const Real dampingDefaultValue = 0.1;

    if (m_damping.getValue().empty())
    {
        VecReal &damping = *m_damping.beginEdit();
        damping.push_back(dampingDefaultValue);
        m_damping.endEdit();
    }

    if (m_angularDamping.getValue().empty())
    {
        VecReal &angDamping = *m_angularDamping.beginEdit();
        angDamping.push_back(dampingDefaultValue);
        m_angularDamping.endEdit();
    }

    const std::string path = external_rest_shape.getValue();

    restMState = NULL;

    if (path.size() > 0)
    {
        this->getContext()->get(restMState ,path);
    }

    if (!restMState)
    {
        useRestMState = false;

        if (path.size() > 0)
        {
            std::cout << "RestShapeSpringsForceField : " << external_rest_shape.getValue() << "not found\n";
        }
    }
    else
    {
        useRestMState = true;

        // std::cout << "RestShapeSpringsForceField : Mechanical state named " << restMState->getName() << " found for RestShapeSpringFF named " << this->getName() << std::endl;
    }

    recomputeIndices();
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::recomputeIndices()
{
    m_indices.clear();
    m_ext_indices.clear();

    for (unsigned int i = 0; i < points.getValue().size(); i++)
        m_indices.push_back(points.getValue()[i]);

    for (unsigned int i = 0; i < external_points.getValue().size(); i++)
        m_ext_indices.push_back(external_points.getValue()[i]);

    m_pivots = pivotPoints.getValue();

    if (m_indices.size()==0)
    {
        //	std::cout << "in RestShapeSpringsForceField no point are defined, default case: points = all points " << std::endl;

        for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
        {
            m_indices.push_back(i);
        }
    }

    if (m_ext_indices.size()==0)
    {
        // std::cout << "in RestShapeSpringsForceField no external_points are defined, default case: points = all points " << std::endl;

        if (useRestMState)
        {
            for (unsigned int i = 0; i < (unsigned)restMState->getSize(); i++)
            {
                m_ext_indices.push_back(i);
            }
        }
        else
        {
            for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
            {
                m_ext_indices.push_back(i);
            }
        }
    }

    if (m_indices.size() > m_ext_indices.size())
    {
        std::cerr << "Error : the dimention of the source and the targeted points are different " << std::endl;
        m_indices.clear();
    }
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addSpringForce(Deriv &f, const Coord &x0, const Coord &x1, const Deriv &v0, const Deriv &v1, const Real &k, const Real &kd)
{
    const Deriv dx = x1 - x0;
    const Deriv dv = v1 - v0;

    Deriv u = dx;
    u.normalize();

    const Real forceIntensity = dx.norm() * k +  dot(u, dv) * kd;

    f -= dx * forceIntensity;
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p0 = *(useRestMState ? restMState->read(core::VecCoordId::position()) : this->mstate->read(core::VecCoordId::restPosition()));
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v1 = v;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v0 = *(useRestMState ? restMState->read(core::VecDerivId::velocity()) : this->mstate->read(core::VecDerivId::null()));

    f1.resize(p1.size());

    if (recompute_indices.getValue())
    {
        recomputeIndices();
    }

    const unsigned int nbIndices = m_indices.size();
    const VecReal &k = stiffness.getValue();
    const VecReal &kd = m_damping.getValue();

    if (k.size() != nbIndices)
    {
        //sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;
        const Real k0 = k[0];

        if (kd.size() != nbIndices)
        {
            const Real kd0 = kd[0];

            for (unsigned int i=0; i < nbIndices; i++)
            {

                const unsigned int index = m_indices[i];
                unsigned int ext_index = m_indices[i];
                if( useRestMState )
                    ext_index = m_ext_indices[i];



                addSpringForce(f1[index], p0[ext_index], p1[index], v0[ext_index], v1[index], k0, kd0);
            }
        }
        else
        {
            for (unsigned int i=0; i < nbIndices; i++)
            {
                const unsigned int index = m_indices[i];
                unsigned int ext_index = m_indices[i];
                if( useRestMState )
                    ext_index = m_ext_indices[i];

                addSpringForce(f1[index], p0[ext_index], p1[index], v0[ext_index], v1[index], k0, kd[i]);
            }
        }
    }
    else
    {
        if (kd.size() != nbIndices)
        {
            const Real kd0 = kd[0];

            for (unsigned int i=0; i < nbIndices; i++)
            {
                const unsigned int index = m_indices[i];
                unsigned int ext_index = m_indices[i];
                if( useRestMState )
                    ext_index = m_ext_indices[i];

                addSpringForce(f1[index], p0[ext_index], p1[index], v0[ext_index], v1[index], k[i], kd0);
            }
        }
        else
        {
            for (unsigned int i=0; i < nbIndices; i++)
            {
                const unsigned int index = m_indices[i];
                unsigned int ext_index = m_indices[i];
                if( useRestMState )
                    ext_index = m_ext_indices[i];

                addSpringForce(f1[index], p0[ext_index], p1[index], v0[ext_index], v1[index], k[i], kd[i]);
            }
        }
    }
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& df, const DataVecDeriv& dx)
{
    //  remove to be able to build in parallel
    // 	const VecIndex& indices = points.getValue();
    // 	const VecReal& k = stiffness.getValue();

    const VecReal& k = stiffness.getValue();

    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;
    double kFactor = mparams->kFactor();

    if (k.size()!= m_indices.size() )
    {
        //sout << "WARNING : stiffness is not defined on each point, first stiffness is used" << sendl;
        const Real k0 = k[0];

        for (unsigned int i=0; i<m_indices.size(); i++)
        {
            df1[m_indices[i]] -=  dx1[m_indices[i]] * k0 * kFactor;
        }
    }
    else
    {
        for (unsigned int i=0; i<m_indices.size(); i++)
        {
            df1[m_indices[i]] -=  dx1[m_indices[i]] * k[i] * kFactor ;
        }
    }
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::draw(const core::visual::VisualParams * /* vparams */ )
{

}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    //      remove to be able to build in parallel
    // 	const VecIndex& indices = points.getValue();

    const VecReal& k = stiffness.getValue();

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    double kFact = mparams->kFactor();

    const int N = Coord::total_size;

    unsigned int curIndex = 0;

    if (k.size()!= m_indices.size() )
    {
        const Real k0 = k[0];
        for (unsigned int index = 0; index < m_indices.size(); index++)
        {
            curIndex = m_indices[index];

            for(int i = 0; i < N; i++)
            {

                //	for (unsigned int j = 0; j < N; j++)
                //	{
                //		mat->add(offset + N * curIndex + i, offset + N * curIndex + j, kFact * k[0]);
                //	}

                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k0);
            }
        }
    }
    else
    {
        for (unsigned int index = 0; index < m_indices.size(); index++)
        {
            curIndex = m_indices[index];

            for(int i = 0; i < N; i++)
            {

                //	for (unsigned int j = 0; j < N; j++)
                //	{
                //		mat->add(offset + N * curIndex + i, offset + N * curIndex + j, kFact * k[curIndex]);
                //	}

                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k[index]);
            }
        }
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL



