/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_DOFBLOCKERLMCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_DOFBLOCKERLMCONSTRAINT_INL

#include <SofaConstraint/DOFBlockerLMConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologySubsetData.inl>


namespace sofa
{

namespace component
{

namespace constraintset
{



// Define TestNewPointFunction
template< class DataTypes>
bool DOFBlockerLMConstraint<DataTypes>::FCTPointHandler::applyTestCreateFunction(unsigned int /*nbPoints*/, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    if (fc)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Define RemovalFunction
template< class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::FCTPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}

template <class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::clearConstraints()
{
    SetIndexArray& _indices = *f_indices.beginEdit();
    _indices.clear();
    f_indices.endEdit();
}

template <class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}


template <class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::init()
{
    core::behavior::LMConstraint<DataTypes,DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    f_indices.createTopologicalEngine(topology, pointHandler);
    f_indices.registerTopologicalData();
}


template<class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::resetConstraint()
{
    core::behavior::LMConstraint<DataTypes,DataTypes>::resetConstraint();
    idxEquations.clear();
}

template<class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /*cParams*/, core::MultiMatrixDerivId cId, unsigned int &cIndex)
{
    if(!idxEquations.empty() ) return;

    using namespace core::objectmodel;
    Data<MatrixDeriv>* dC = cId[this->constrainedObject1].write();
    helper::WriteAccessor<Data<MatrixDeriv> > c = *dC;

    const SetIndexArray &indices = f_indices.getValue();
    const helper::vector<Deriv> &axis=BlockedAxis.getValue();
    idxEquations.resize(indices.size());
    unsigned int numParticle=0;

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it, ++numParticle)
    {
        const unsigned int index=*it;
        for (unsigned int i=0; i<axis.size(); ++i)
        {
            c->writeLine(cIndex).addCol(index,axis[i]);
            idxEquations[numParticle].push_back(cIndex++);
        }
        this->constrainedObject1->forceMask.insertEntry(index);
    }


}

template<class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::writeConstraintEquations(unsigned int& lineNumber, core::MultiVecId id, ConstOrder Order)
{
    using namespace core;
    using namespace core::objectmodel;
    //We don't constrain the Position, only the velocities and accelerations
    if (idxEquations.empty() ||
        Order==core::ConstraintParams::POS) return;

    const SetIndexArray & indices = f_indices.getValue();
    const helper::vector<SReal> &factor=factorAxis.getValue();

    for (unsigned int numParticle=0; numParticle<indices.size(); ++numParticle)
    {
        for (unsigned int i=0; i<idxEquations[numParticle].size(); ++i)
        {
            core::behavior::ConstraintGroup *constraint = this->addGroupConstraint(Order);
            SReal correction=0;
            switch(Order)
            {
            case core::ConstraintParams::ACC :
            case core::ConstraintParams::VEL :
            {
                ConstVecId v1 = id.getId(this->constrainedObject1);
                correction = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(idxEquations[numParticle][i],v1);
                break;
            }
            default: break;
            };
            if (!factor.empty())
            {
                if (i < factor.size()) correction*=factor[i];
                else                   correction*=factor.back();
            }
            constraint->addConstraint( lineNumber, idxEquations[numParticle][i], -correction);
        }
    }
}

template <class DataTypes>
void DOFBlockerLMConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowForceFields()) return;
    const VecCoord& x =this->constrainedObject1->read(core::ConstVecCoordId::position())->getValue();

    const SetIndexArray & indices = f_indices.getValue();

    for (SetIndexArray::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        unsigned int index=(*it);
        Coord pos=x[index];
        defaulttype::Vector3 position;
        DataTypes::get(position[0], position[1], position[2], pos);
        glColor3f(1,1,0);
        const helper::vector<Deriv>& axis=BlockedAxis.getValue();
        for (unsigned int i=0; i<axis.size(); ++i)
        {
            defaulttype::Vector3 direction;
            DataTypes::get(direction[0], direction[1], direction[2],axis[i]);
            helper::gl::Axis::draw(position,position+direction*showSizeAxis.getValue(),
                    showSizeAxis.getValue()*0.03);
        }
    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif


