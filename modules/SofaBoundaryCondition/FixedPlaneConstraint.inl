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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_INL

#include <SofaBoundaryCondition/FixedPlaneConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::FixedPlaneConstraint()
    : direction( initData(&direction,"direction","normal direction of the plane"))
    , dmin( initData(&dmin,(Real)0,"dmin","Minimum plane distance from the origin"))
    , dmax( initData(&dmax,(Real)0,"dmax","Maximum plane distance from the origin") )
    , indices( initData(&indices,"indices","Indices of the fixed points"))
{
    selectVerticesFromPlanes=false;
    pointHandler = new FCPointHandler(this, &indices);
}

template <class DataTypes>
FixedPlaneConstraint<DataTypes>::~FixedPlaneConstraint()
{
    if (pointHandler)
        delete pointHandler;
}

// Define TestNewPointFunction
template< class DataTypes>
bool FixedPlaneConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
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

// Matrix Integration interface
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get(mparams));
    if(r)
    {
        //sout << "applyConstraint in Matrix with offset = " << offset << sendl;
        //cerr<<"FixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset) is called "<<endl;
        const unsigned int N = Deriv::size();
        const SetIndexArray & ind = indices.getValue();

        // IMplement plane constraint only when the direction is along the coordinates directions
		// TODO : generalize projection to any direction
		 Coord dir=direction.getValue();

        for (SetIndexArray::const_iterator it = ind.begin(); it != ind.end(); ++it)
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<N; ++c)
				if (dir[c]!=0.0)
                r.matrix->clearRowCol(r.offset + N * (*it) + c);
            // Set Fixed Vertex
            for (unsigned int c=0; c<N; ++c)
				if (dir[c]!=0.0)
                r.matrix->set(r.offset + N * (*it) + c, r.offset + N * (*it) + c, 1.0);
        }
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, defaulttype::BaseVector* vect, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    int o = matrix->getGlobalOffset(this->mstate.get(mparams));
    if (o >= 0)
    {
        unsigned int offset = (unsigned int)o;
		// IMplement plane constraint only when the direction is along the coordinates directions
		// TODO : generalize projection to any direction
		Coord dir=direction.getValue();

        const unsigned int N = Deriv::size();

      


        const SetIndexArray & ind = indices.getValue();
        for (SetIndexArray::const_iterator it = ind.begin(); it != ind.end(); ++it)
        {
            for (unsigned int c=0; c<N; ++c)
				if (dir[c]!=0.0)
                vect->clear(offset + N * (*it) + c);
        }
    }
}

// Define RemovalFunction
template< class DataTypes>
void FixedPlaneConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::addConstraint(int index)
{
    indices.beginEdit()->push_back(index);
    indices.endEdit();
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::removeConstraint(int index)
{
    removeValue(*indices.beginEdit(),(unsigned int)index);
    indices.endEdit();
}

// -- Mass interface


template <class DataTypes> template <class DataDeriv>
void FixedPlaneConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/, DataDeriv& res)
{
    Coord dir=direction.getValue();

    for (helper::vector< unsigned int >::const_iterator it = this->indices.getValue().begin(); it != this->indices.getValue().end(); ++it)
    {
        /// only constraint one projection of the displacement to be zero
        res[*it]-= dir*dot(res[*it],dir);
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(mparams, res.wref());
}

/// project dx to constrained space (dx models a velocity)
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*vData*/)
{

}

/// project x to constrained space (x models a position)
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{

}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned /*offset*/ )
{
    // clears the rows and columns associated with constrained particles
    unsigned blockSize = DataTypes::deriv_total_size;

    for(SetIndexArray::const_iterator it= indices.getValue().begin(), iend=indices.getValue().end(); it!=iend; it++ )
    {
        M->clearRowsCols((*it) * blockSize,(*it+1) * (blockSize) );
    }

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        projectResponseT<MatrixDerivRowType>(mparams, rowIt.row());
        ++rowIt;
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::setDirection(Coord dir)
{
    if (dir.norm2()>0)
    {
        direction.setValue(dir);
    }
}

template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::selectVerticesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    unsigned int i;
    for(i=0; i<x.size(); ++i)
    {
        if (isPointInPlane(x[i]))
            addConstraint(i);
    }

}
template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    /// test that dmin or dmax are different from zero
    if (dmin.getValue()!=dmax.getValue())
        selectVerticesFromPlanes=true;

    if (selectVerticesFromPlanes)
        selectVerticesAlongPlane();

    // Initialize functions and parameters
    indices.createTopologicalEngine(topology, pointHandler);
    indices.registerTopologicalData();

}


template <class DataTypes>
void FixedPlaneConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    vparams->drawTool()->disableLighting();

    sofa::helper::vector<sofa::defaulttype::Vector3> points;
    for (helper::vector< unsigned int >::const_iterator it = this->indices.getValue().begin(); it != this->indices.getValue().end(); ++it)
    {
        points.push_back({x[*it][0], x[*it][1], x[*it][2]});
    }

    vparams->drawTool()->drawPoints(points, 10, {1,1.0,0.5,1});
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
