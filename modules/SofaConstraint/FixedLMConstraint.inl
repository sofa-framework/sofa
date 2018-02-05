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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_FIXEDLMCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINTSET_FIXEDLMCONSTRAINT_INL

#include <SofaConstraint/FixedLMConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
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
bool FixedLMConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int /*nbPoints*/, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
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
void FixedLMConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}

template <class DataTypes>
void FixedLMConstraint<DataTypes>::clearConstraints()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void FixedLMConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void FixedLMConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}


template <class DataTypes>
void FixedLMConstraint<DataTypes>::initFixedPosition()
{
    this->restPosition.clear();
    const VecCoord& x =this->constrainedObject1->read(core::ConstVecCoordId::position())->getValue();
    const SetIndexArray & indices = this->f_indices.getValue();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        unsigned int index=*it;
        this->restPosition.insert(std::make_pair(index, x[index]));
    }
}

template <class DataTypes>
void FixedLMConstraint<DataTypes>::init()
{
    core::behavior::LMConstraint<DataTypes,DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    f_indices.createTopologicalEngine(topology, pointHandler);
    f_indices.registerTopologicalData();


    X[0]=1; X[1]=0; X[2]=0;
    Y[0]=0; Y[1]=1; Y[2]=0;
    Z[0]=0; Z[1]=0; Z[2]=1;

    initFixedPosition();

}



template<class DataTypes>
void FixedLMConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams* /* cParams*/, core::MultiMatrixDerivId cId, unsigned int &cIndex)
{
    using namespace core::objectmodel;
    Data<MatrixDeriv>* dC = cId[this->constrainedObject1].write();
    helper::WriteAccessor<Data<MatrixDeriv> > c = *dC;
    idxX.clear();
    idxY.clear();
    idxZ.clear();
    const SetIndexArray &indices = f_indices.getValue();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        const unsigned int index=*it;

        //Constraint degree of freedom along X direction
        c->writeLine(cIndex).addCol(index,X);
        idxX.push_back(cIndex++);

        //Constraint degree of freedom along X direction
        c->writeLine(cIndex).addCol(index,Y);
        idxY.push_back(cIndex++);

        //Constraint degree of freedom along Z direction
        c->writeLine(cIndex).addCol(index,Z);
        idxZ.push_back(cIndex++);

        this->constrainedObject1->forceMask.insertEntry(index);
    }
}

template<class DataTypes>
void FixedLMConstraint<DataTypes>::writeConstraintEquations(unsigned int& lineNumber, core::MultiVecId id, ConstOrder Order)
{
    using namespace core;
    using namespace core::objectmodel;
    const SetIndexArray & indices = f_indices.getValue();

    unsigned int counter=0;
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it,++counter)
    {
        const unsigned int index = *it;

        core::behavior::ConstraintGroup *constraint = this->addGroupConstraint(Order);
        SReal correctionX=0,correctionY=0,correctionZ=0;
        switch(Order)
        {
        case core::ConstraintParams::ACC :
        case core::ConstraintParams::VEL :
        {
            ConstVecId v1 = id.getId(this->constrainedObject1);
            correctionX = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(idxX[counter],v1);
            correctionY = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(idxY[counter],v1);
            correctionZ = this->constrainedObject1->getConstraintJacobianTimesVecDeriv(idxZ[counter],v1);
            break;
        }
        case core::ConstraintParams::POS :
        case core::ConstraintParams::POS_AND_VEL :
        {
            ConstVecId xid = id.getId(this->constrainedObject1);
            helper::ReadAccessor<Data<VecCoord> > x = *this->constrainedObject1->read((ConstVecCoordId)xid);

            //If a new particle has to be fixed, we add its current position as rest position
            if (restPosition.find(index) == this->restPosition.end())
            {
                restPosition.insert(std::make_pair(index, x[index]));
            }


            Coord v=x[index]-restPosition[index];
            correctionX=v[0];
            correctionY=v[1];
            correctionZ=v[2];
            break;
        }
        };

        constraint->addConstraint( lineNumber, idxX[counter], -correctionX);
        constraint->addConstraint( lineNumber, idxY[counter], -correctionY);
        constraint->addConstraint( lineNumber, idxZ[counter], -correctionZ);
    }
}



template <class DataTypes>
void FixedLMConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x =this->constrainedObject1->read(core::ConstVecCoordId::position())->getValue();
    //serr<<"FixedLMConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<sendl;

    const SetIndexArray & indices = f_indices.getValue();

    std::vector< defaulttype::Vector3 > points;
    defaulttype::Vector3 point;
    //serr<<"FixedLMConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
    for (SetIndexArray::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        point = DataTypes::getCPos(x[*it]);
        points.push_back(point);
    }
    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        vparams->drawTool()->drawPoints(points, 10, defaulttype::Vec<4,float>(1,0.5,0.5,1));
    }
    else
    {
        vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), defaulttype::Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }
}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void FixedLMConstraint<defaulttype::Rigid3dTypes >::draw(const core::visual::VisualParams* vparams);
#endif
#ifndef SOFA_DOUBLE
template <>
void FixedLMConstraint<defaulttype::Rigid3fTypes >::draw(const core::visual::VisualParams* vparams);
#endif


} // namespace constraintset

} // namespace component

} // namespace sofa

#endif


