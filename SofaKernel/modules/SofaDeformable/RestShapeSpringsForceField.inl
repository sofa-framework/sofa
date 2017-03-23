/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL

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
    , pivotPoints(initData(&pivotPoints, "pivot_points", "global pivot points used when translations instead of the rigid mass centers"))
    , external_rest_shape(initData(&external_rest_shape, "external_rest_shape", "rest_shape can be defined by the position of an external Mechanical State"))
    , external_points(initData(&external_points, "external_points", "points from the external Mechancial State that define the rest shape springs"))
    , recompute_indices(initData(&recompute_indices, false, "recompute_indices", "Recompute indices (should be false for BBOX)"))
    , d_drawSpring(initData(&d_drawSpring, false, "drawSpring", "draw Spring"))
    , d_drawSpringLengthThreshold(initData(&d_drawSpringLengthThreshold, (Real)0.1, "drawSpringLengthThreshold", "Display : When spring length is under this threshold a sphere is displayed instead of a line"))
    , d_springColor(initData(&d_springColor, sofa::defaulttype::Vec4f(0.f,1.f,0.f,1.f), "springColor", "Display : spring color"))
    , d_springSphereColor(initData(&d_springSphereColor, sofa::defaulttype::Vec4f(1.f,.5f,0.5f,1.f), "springSphereColor", "Display : spring sphere color (used when springs are used as fixed constraint)"))
    , d_springSphereRadius(initData(&d_springSphereRadius, (Real)0.2, "springSphereRadius", "Display : spring sphere radius (used when springs are used as fixed constraint)"))
    , restMState(NULL)
    , d_useRestMState(initData(&d_useRestMState, "useRestMState", "An external MechanicalState is used as rest reference."))
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

    const std::string path = external_rest_shape.getValue();

    restMState = NULL;

    if (path.size() > 0)
    {
        this->getContext()->get(restMState ,path);
    }

    d_useRestMState.setValue(restMState != NULL);
    d_useRestMState.setReadOnly(true);

    if (!d_useRestMState.getValue() && (path.size() > 0))
    {
        serr << "RestShapeSpringsForceField : " << external_rest_shape.getValue() << " not found" << sendl;
    }

    recomputeIndices();

#ifdef SOFA_HAVE_EIGEN2
    core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    assert(state);
    matS.resize(state->getMatrixSize(),state->getMatrixSize());
    lastUpdatedStep = -1.0;
#endif
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::reinit()
{
    if (stiffness.getValue().empty())
    {
        std::cout << "RestShapeSpringsForceField : No stiffness is defined, assuming equal stiffness on each node, k = 100.0 " << std::endl;

        VecReal stiffs;
        stiffs.push_back(100.0);
        stiffness.setValue(stiffs);
    }
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::recomputeIndices()
{
    m_indices.clear();
    m_ext_indices.clear();

    for (unsigned int i = 0; i < points.getValue().size(); i++)
    {
        m_indices.push_back(points.getValue()[i]);
    }

    for (unsigned int i = 0; i < external_points.getValue().size(); i++)
    {
        m_ext_indices.push_back(external_points.getValue()[i]);
    }

    if (m_indices.empty())
    {
        // no point are defined, default case: points = all points

        for (unsigned int i = 0; i < (unsigned)this->mstate->getSize(); i++)
        {
            m_indices.push_back(i);
        }
    }

    if (m_ext_indices.empty())
    {
        if (!d_useRestMState.getValue())
        {
            for (unsigned int i = 0; i < m_indices.size(); i++)
            {
                m_ext_indices.push_back(m_indices[i]);
            }
        }
        else
        {
            for (unsigned int i = 0; i < getExtPosition()->getValue().size(); i++)
            {
                m_ext_indices.push_back(i);
            }
        }
    }

    if (!checkOutOfBoundsIndices())
    {
        serr << "RestShapeSpringsForceField is not activated." << sendl;
        m_indices.clear();
    }

    m_pivots = pivotPoints.getValue();
}


template<class DataTypes>
bool RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices()
{
    if (!checkOutOfBoundsIndices(m_indices, this->mstate->getSize()))
    {
        serr << "RestShapeSpringsForceField : Out of Bounds m_indices detected. ForceField is not activated." << sendl;
        return false;
    }

    if (!checkOutOfBoundsIndices(m_ext_indices, getExtPosition()->getValue().size()))
    {
        serr << "RestShapeSpringsForceField : Out of Bounds m_ext_indices detected. ForceField is not activated." << sendl;
        return false;
    }

    if (m_indices.size() != m_ext_indices.size())
    {
        serr << "RestShapeSpringsForceField : Dimensions of the source and the targeted points are different. ForceField is not activated." << sendl;
        return false;
    }

    return true;
}


template<class DataTypes>
bool RestShapeSpringsForceField<DataTypes>::checkOutOfBoundsIndices(const VecIndex &indices, const unsigned int dimension)
{
    for (unsigned int i = 0; i < indices.size(); i++)
    {
        if (indices[i] >= dimension)
        {
            return false;
        }
    }

    return true;
}


template<class DataTypes>
const typename RestShapeSpringsForceField<DataTypes>::DataVecCoord* RestShapeSpringsForceField<DataTypes>::getExtPosition() const
{
    return (d_useRestMState.getValue() ? restMState->read(core::VecCoordId::position()) : this->mstate->read(core::VecCoordId::restPosition()));
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
    sofa::helper::WriteAccessor< DataVecDeriv > f1 = f;
    sofa::helper::ReadAccessor< DataVecCoord > p1 = x;
    sofa::helper::ReadAccessor< DataVecCoord > p0 = *getExtPosition();

    f1.resize(p1.size());

    if (recompute_indices.getValue())
    {
        recomputeIndices();
    }

    const VecReal& k = stiffness.getValue();

    if (k.size() != m_indices.size())
    {
        // Stiffness is not defined on each point, first stiffness is used
        const Real k0 = k[0];

        for (unsigned int i=0; i<m_indices.size(); i++)
        {
            const unsigned int index = m_indices[i];
            const unsigned int ext_index = m_ext_indices[i];

            Deriv dx = p1[index] - p0[ext_index];
            f1[index] -=  dx * k0 ;
        }
    }
    else
    {
        for (unsigned int i=0; i<m_indices.size(); i++)
        {
            const unsigned int index = m_indices[i];
            const unsigned int ext_index = m_ext_indices[i];

            Deriv dx = p1[index] - p0[ext_index];
            f1[index] -=  dx * k[i];
        }
    }
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< DataVecDeriv > df1 = df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());


    const VecReal& k = stiffness.getValue();

    if (k.size() != m_indices.size())
    {
        // Stiffness is not defined on each point, first stiffness is used
        const Real k0 = k[0] * (Real)kFactor;

        for (unsigned int i=0; i<m_indices.size(); i++)
        {
            df1[m_indices[i]] -=  dx1[m_indices[i]] * k0;
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
void RestShapeSpringsForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    //      remove to be able to build in parallel
    // 	const VecIndex& indices = points.getValue();
    // 	const VecReal& k = stiffness.getValue();   
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    const VecReal& k = stiffness.getValue();


    const int N = Coord::total_size;

    unsigned int curIndex = 0;

    if (k.size() != m_indices.size())
    {
        const Real k0 = -k[0] * (Real)kFact;

        for (unsigned int index = 0; index < m_indices.size(); index++)
        {
            curIndex = m_indices[index];

            for(int i = 0; i < N; i++)
            {
                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, k0);
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
                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k[index]);
            }
        }
    }
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::addSubKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix, const helper::vector<unsigned> & addSubIndex )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;
    Real kFact = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    const VecReal& k = stiffness.getValue();
    const int N = Coord::total_size;

    unsigned int curIndex = 0;

    if (k.size() != m_indices.size())
    {
        const Real k0 = -k[0] * (Real)kFact;

        for (unsigned int index = 0; index < m_indices.size(); index++)
        {
            curIndex = m_indices[index];

            if (std::find(addSubIndex.begin(), addSubIndex.end(), curIndex) == addSubIndex.end())
                continue;

            for(int i = 0; i < N; i++)
            {
                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, k0);
            }
        }
    }
    else
    {
        for (unsigned int index = 0; index < m_indices.size(); index++)
        {
            curIndex = m_indices[index];

            if (std::find(addSubIndex.begin(), addSubIndex.end(), curIndex) == addSubIndex.end())
                continue;

            for(int i = 0; i < N; i++)
            {
                mat->add(offset + N * curIndex + i, offset + N * curIndex + i, -kFact * k[index]);
            }
        }
    }
}

template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::updateForceMask()
{
    for (unsigned int i=0; i<m_indices.size(); i++)
        this->mstate->forceMask.insertEntry(m_indices[i]);
}


template<class DataTypes>
void RestShapeSpringsForceField<DataTypes>::draw(const core::visual::VisualParams *vparams)
{
    if (!vparams->displayFlags().getShowForceFields())
        return;

    if (!d_drawSpring.getValue())
        return;

    sofa::helper::ReadAccessor< DataVecCoord > p0 = *getExtPosition();
    sofa::helper::ReadAccessor< DataVecCoord > p = this->mstate->read(core::VecCoordId::position());

    const VecIndex& indices = m_indices;
    const VecIndex& ext_indices = m_ext_indices;

    helper::vector< defaulttype::Vector3 > lines;
    helper::vector< defaulttype::Vector3 > points;
    sofa::defaulttype::Vector3 point1, point2;

    for (unsigned int i = 0; i < indices.size(); i++)
    {
        point1 = DataTypes::getCPos(p[indices[i]]);
        point2 = DataTypes::getCPos(p0[ext_indices[i]]);

        if ((point2-point1).norm() > d_drawSpringLengthThreshold.getValue())
        {
            lines.push_back(point1);
            lines.push_back(point2);
        }
        else
        {
            points.push_back(point1);
        }
    }

    vparams->drawTool()->drawLines(lines, 5.f, d_springColor.getValue());
    vparams->drawTool()->drawSpheres(points, (float)d_springSphereRadius.getValue(), d_springSphereColor.getValue());
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGFORCEFIELD_INL



