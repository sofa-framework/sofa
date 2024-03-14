/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/mechanicalload/ConstantForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>

#include <math.h>
#include <cassert>
#include <iostream>
#include <numeric>

namespace sofa::component::mechanicalload
{

template<class DataTypes>
ConstantForceField<DataTypes>::ConstantForceField()
    : d_indices(initData(&d_indices, "indices", "indices where the forces are applied"))
    , d_indexFromEnd(initData(&d_indexFromEnd,false,"indexFromEnd", "Concerned DOFs indices are numbered from the end of the MState DOFs vector. (default=false)"))
    , d_forces(initData(&d_forces, "forces", "applied forces at each point"))
    , d_totalForce(initData(&d_totalForce, "totalForce", "total force for all points, will be distributed uniformly over points"))
    , d_showArrowSize(initData(&d_showArrowSize, 0_sreal, "showArrowSize", "Size of the drawn arrows (0->no arrows, sign->direction of drawing. (default=0)"))
    , d_color(initData(&d_color, sofa::type::RGBAColor(0.2f,0.9f,0.3f,1.0f), "showColor", "Color for object display (default: [0.2,0.9,0.3,1.0])"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_systemSize(0)
{
    d_forces.setGroup("Force info");
    d_totalForce.setGroup("Force info");

    d_showArrowSize.setGroup("Visualization");
    d_color.setGroup("Visualization");

    sofa::core::objectmodel::Base::addUpdateCallback("updateFromIndices", {&d_indices}, [this](const core::DataTracker& )
    {
        msg_info() << "call back update: from indices";
        return updateFromIndices();
    }, {});

    sofa::core::objectmodel::Base::addUpdateCallback("updateFromForcesVector", {&d_forces, &d_indices}, [this](const core::DataTracker& )
    {
        if(!m_isTotalForceUsed)
        {
            msg_info() << "dataInternalUpdate: data forces has changed";
            return updateFromForcesVector();
        }
        else
        {
            msg_info() << "totalForce data is initially used, the callback associated with the forces vector is skipped";
            return updateFromTotalForce();
        }
    }, {});

    sofa::core::objectmodel::Base::addUpdateCallback("updateFromTotalForce", {&d_totalForce, &d_indices}, [this](const core::DataTracker& )
    {
        if(m_isTotalForceUsed)
        {
            msg_info() << "dataInternalUpdate: data totalForce has changed";
            return updateFromTotalForce();
        }
        else
        {
            msg_info() << "forces data is initially used, the callback associated with the totalForce is skipped";
            return updateFromForcesVector();
        }
    }, {});
}


template<class DataTypes>
void ConstantForceField<DataTypes>::init()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
       
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    // temprory pointer to topology
    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";
        
        // Initialize functions and parameters for topology data and handler
        d_indices.createTopologyHandler(_topology);

        m_systemSize = _topology->getNbPoints();
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        const core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
        m_systemSize = state->getSize();
    }



    /// Check on data isSet()
    if (d_forces.isSet())
    {
        if(d_totalForce.isSet())
        {
            msg_warning() <<"Both data \'forces\' and \'totalForce\' cannot be used simultaneously, please set only one of them to remove this warning";
        }

        m_isTotalForceUsed = false;
        d_totalForce.setReadOnly(true);

        msg_info() << "Input vector forces is used for initialization";
    }
    else if (d_totalForce.isSet())
    {
        m_isTotalForceUsed = true;
        d_forces.setReadOnly(true);

        msg_info() << "Input totalForce is used for initialization";
    }
    else
    {
        msg_error() << "No input force has been set. Please define one of both Data: " << d_forces.getName() << " or " << d_totalForce.getName();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (!d_indices.isSet() || d_indices.getValue().size() == 0)
    {
        // initialize with all indices
        auto indicesEdit = sofa::helper::getWriteAccessor(d_indices);
        indicesEdit.clear();
        indicesEdit.resize(m_systemSize);
        std::iota (std::begin(indicesEdit), std::end(indicesEdit), 0);
    }

    // init from ForceField
    Inherit::init();

    // if all init passes, component is valid
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState ConstantForceField<DataTypes>::updateFromIndices()
{
    const VecIndex & indices = d_indices.getValue();
    const size_t indicesSize = indices.size();

    // check size of vector indices
    if( indicesSize > m_systemSize )
    {
        msg_error() << "Size mismatch: indices > system size";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
    else if( indicesSize==0 )
    {
        msg_error() << "Size of input vector indices is zero";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }

    // check each indice of the vector
    for(Size i=0; i<indicesSize; i++)
    {
        if( indices[i] > m_systemSize )
        {
            msg_error() << "Indices incorrect: indice["<< i <<"] = "<< indices[i] <<" exceeds system size";
            return sofa::core::objectmodel::ComponentState::Invalid;
        }
    }

    return sofa::core::objectmodel::ComponentState::Valid;
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState ConstantForceField<DataTypes>::updateFromForcesVector()
{
    const VecDeriv &forces = d_forces.getValue();
    if( checkForces(forces) )
    {
        if(forces.size() == 1)
        {
            // if only one Deriv force is given as input, replicate this force on all indices
            return computeForceFromSingleForce(forces[0]);
        }
        else
        {
            return computeForceFromForcesVector(forces);
        }
    }
    else
    {
        msg_error() << " Invalid given vector forces";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState ConstantForceField<DataTypes>::updateFromTotalForce()
{
    const Deriv &totalForce = d_totalForce.getValue();
    if( checkForce(totalForce) )
    {
        return computeForceFromTotalForce(totalForce);
    }
    else
    {
        msg_error() << " Invalid given totalForce";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
}


template<class DataTypes>
bool ConstantForceField<DataTypes>::checkForce(const Deriv& force)
{
    for (typename Deriv::Size i=0; i<Deriv::spatial_dimensions; ++i)
    {
        if( std::isnan(force[i]) )
        {
            return false;
        }
    }
    return true;
}


template<class DataTypes>
bool ConstantForceField<DataTypes>::checkForces(const VecDeriv& forces)
{
    for (auto&& i : forces)
    {
        if(! checkForce(i) )
        {
            return false;
        }
    }
    return true;
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState ConstantForceField<DataTypes>::computeForceFromForcesVector(const VecDeriv &forces)
{
    const size_t indicesSize = d_indices.getValue().size();
    Deriv totalForce;

    totalForce.clear();

    if( indicesSize!=forces.size() )
    {
        msg_error() << "Impossible to use the vector forces since its size mismatches with indices size";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
    else
    {
        for(Size i=0; i<indicesSize; i++)
        {
            totalForce += forces[i];
        }
    }
    d_totalForce.setValue(totalForce);
    return sofa::core::objectmodel::ComponentState::Valid;
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState ConstantForceField<DataTypes>::computeForceFromSingleForce(const Deriv singleForce)
{
    const VecIndex & indices = d_indices.getValue();
    const size_t indicesSize = indices.size();
    sofa::helper::WriteAccessor<DataVecDeriv> forces = d_forces;

    forces.clear();
    forces.resize(indicesSize);

    for(Size i=0; i<indicesSize; i++)
    {
        forces[i] = singleForce;
    }

    d_totalForce.setValue(singleForce*(static_cast<Real>(indicesSize)));

    return sofa::core::objectmodel::ComponentState::Valid;
}


template<class DataTypes>
sofa::core::objectmodel::ComponentState ConstantForceField<DataTypes>::computeForceFromTotalForce(const Deriv &totalForce)
{
    const size_t indicesSize = d_indices.getValue().size();
    sofa::helper::WriteAccessor<DataVecDeriv> forces = d_forces;

    if( indicesSize!=0 )
    {
        Deriv singleForce;
        singleForce = totalForce / (static_cast<Real>(indicesSize));

        forces.clear();
        forces.resize(indicesSize);

        for(Size i=0; i<indicesSize; i++)
        {
            forces[i] = singleForce;
        }
        return sofa::core::objectmodel::ComponentState::Valid;
    }
    else
    {
        msg_error() << "Impossible to compute force from totalForce since vector indices size is zero";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addForce(const core::MechanicalParams* params, DataVecDeriv& f, const DataVecCoord& x1, const DataVecDeriv& v1)
{
    if(!this->isComponentStateValid())
        return;

    SOFA_UNUSED(params);
    SOFA_UNUSED(x1);
    SOFA_UNUSED(v1);

    sofa::helper::WriteAccessor<DataVecDeriv> _f1 = f;
    const VecIndex& indices = d_indices.getValue();
    const VecDeriv& forces = d_forces.getValue();

    const size_t indicesSize = indices.size();
    m_systemSize = _f1.size();

    if (!d_indexFromEnd.getValue())
    {
        for (size_t i=0; i<indicesSize; i++)
        {
            _f1[indices[i]] += forces[i];
        }
    }
    else
    {
        for (size_t i=0; i<indicesSize; i++)
        {
            _f1[m_systemSize - indices[i] - 1] += forces[i];
        }
    }
}

template <class DataTypes>
SReal ConstantForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* params, const DataVecCoord& x) const
{
    SOFA_UNUSED(params);
    const VecIndex& indices = d_indices.getValue();
    const VecDeriv& f = d_forces.getValue();
    const VecCoord& _x = x.getValue();
    SReal e = 0;
    unsigned int i = 0;

    if (!d_indexFromEnd.getValue())
    {
        for (; i<indices.size(); i++)
        {
            e -= f[i] * _x[indices[i]];
        }
    }
    else
    {
        for (; i < indices.size(); i++)
        {
            e -= f[i] * _x[_x.size() - indices[i] -1];
        }
    }

    return e;
}


template <class DataTypes>
void ConstantForceField<DataTypes>::setForce(unsigned i, const Deriv& force)
{
    if(m_isTotalForceUsed)
    {
        msg_warning() << "\'forces\' vector is modified using setForce() while totalMass is initially used. "
                      << "Now the 'forces\' vector is used.";
        m_isTotalForceUsed = false;
    }

    auto indices = sofa::helper::getWriteAccessor(d_indices);
    sofa::helper::WriteAccessor<DataVecDeriv> f = d_forces;

    indices.push_back(i);
    f.push_back( force );
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx)
{
    // Derivative of a constant force is null, no need to compute addKToMatrix nor addDForce
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(d_df);
    SOFA_UNUSED(d_dx);
    mparams->setKFactorUsed(true);
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix * mat, SReal k, unsigned int & offset)
{
    // Derivative of a constant force is null, no need to compute addKToMatrix nor addDForce
    SOFA_UNUSED(mat);
    SOFA_UNUSED(k);
    SOFA_UNUSED(offset);
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, SReal kFact)
{
    // Derivative of a constant force is null, no need to compute addKToMatrix nor addDForce
    SOFA_UNUSED(matrix);
    SOFA_UNUSED(kFact);
}

template <class DataTypes>
void ConstantForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    SOFA_UNUSED(matrix);
}

template<class DataTypes>
void ConstantForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(!this->isComponentStateValid())
        return;

    const SReal aSC = d_showArrowSize.getValue();

    if (!vparams->displayFlags().getShowForceFields() || (aSC <= 0.0))
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecIndex& indices = d_indices.getValue();
    const VecDeriv& f = d_forces.getValue();
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if( fabs(aSC)<1.0e-10 )
    {
        std::vector<type::Vec3> points;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Real xx = 0.0, xy = 0.0, xz = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

            if (!d_indexFromEnd.getValue())
            {
                if (indices[i] < x.size())
                {
                    DataTypes::get(xx, xy, xz, x[indices[i]]);
                }
                else
                {
                    msg_error() << "Draw: error in indices values";
                }
            }
            else
            {
                if ((x.size() - indices[i] - 1) < x.size() && (x.size() - indices[i] - 1) >= 0)
                {
                    DataTypes::get(xx, xy, xz, x[x.size() - indices[i] - 1]);
                }
                else
                {
                    msg_error() << "Draw: error in indices values";
                }
            }

            DataTypes::get(fx,fy,fz, f[i] );
            points.push_back(type::Vec3(xx, xy, xz ));
            points.push_back(type::Vec3(xx+fx, xy+fy, xz+fz ));
        }
        vparams->drawTool()->drawLines(points, 2, sofa::type::RGBAColor::green());
    }
    else
    {

        vparams->drawTool()->setLightingEnabled(true);
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Real xx = 0.0, xy = 0.0, xz = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

            if (!d_indexFromEnd.getValue())
            {
                if (indices[i] < x.size())
                {
                    DataTypes::get(xx, xy, xz, x[indices[i]]);
                }
                else
                {
                    msg_error() << "Draw: error in indices values";
                }
            }
            else
            {
                if ((x.size() - indices[i] - 1) < x.size() && (x.size() - indices[i] - 1) >= 0)
                {
                    DataTypes::get(xx, xy, xz, x[x.size() - indices[i] - 1]);
                }
                else
                {
                    msg_error() << "Draw: error in indices values";
                }
            }

            DataTypes::get(fx,fy,fz, f[i] );

            type::Vec3 p1( xx, xy, xz);
            type::Vec3 p2( aSC*fx+xx, aSC*fy+xy, aSC*fz+xz );

            const float norm = static_cast<float>((p2-p1).norm());

            if( aSC > 0.0)
            {
                vparams->drawTool()->drawArrow(p1,p2, norm/20.0f, d_color.getValue());
            }
            else
            {
                vparams->drawTool()->drawArrow(p2,p1, norm/20.0f, d_color.getValue());
            }
        }
    }
}

} // namespace sofa::component::mechanicalload
