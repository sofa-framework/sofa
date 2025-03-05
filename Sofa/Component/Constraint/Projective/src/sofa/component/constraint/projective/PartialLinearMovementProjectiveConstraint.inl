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

#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/component/constraint/projective/PartialLinearMovementProjectiveConstraint.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/RGBAColor.h>
#include <iostream>
#include <sofa/type/vector_algorithm.h>


namespace sofa::component::constraint::projective
{

template <class DataTypes>
PartialLinearMovementProjectiveConstraint<DataTypes>::PartialLinearMovementProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , data(std::make_unique<PartialLinearMovementProjectiveConstraintInternalData<DataTypes>>())
    , d_indices( initData(&d_indices,"indices","Indices of the constrained points") )
    , d_keyTimes(  initData(&d_keyTimes,"keyTimes","key times for the movements") )
    , d_keyMovements(  initData(&d_keyMovements,"movements","movements corresponding to the key times") )
    , d_showMovement( initData(&d_showMovement, (bool)false, "showMovement", "Visualization of the movement to be applied to constrained dofs."))
    , d_linearMovementBetweenNodesInIndices( initData(&d_linearMovementBetweenNodesInIndices, (bool)false, "linearMovementBetweenNodesInIndices", "Take into account the linear movement between the constrained points"))
    , d_mainIndice( initData(&d_mainIndice, "mainIndice", "The main indice node in the list of constrained nodes, it defines how to apply the linear movement between this constrained nodes "))
    , d_minDepIndice( initData(&d_minDepIndice, "minDepIndice", "The indice node in the list of constrained nodes, which is imposed the minimum displacement "))
    , d_maxDepIndice( initData(&d_maxDepIndice, "maxDepIndice", "The indice node in the list of constrained nodes, which is imposed the maximum displacement "))
    , d_imposedDisplacmentOnMacroNodes(  initData(&d_imposedDisplacmentOnMacroNodes,"imposedDisplacmentOnMacroNodes","The imposed displacement on macro nodes") )
    , d_X0 ( initData ( &d_X0, Real(0.0),"X0","Size of specimen in X-direction" ) )
    , d_Y0 ( initData ( &d_Y0, Real(0.0),"Y0","Size of specimen in Y-direction" ) )
    , d_Z0 ( initData ( &d_Z0, Real(0.0),"Z0","Size of specimen in Z-direction" ) )
    , d_movedDirections( initData(&d_movedDirections,"movedDirections","Defines the directions in which the particles are moved: true (or 1) for fixed, false (or 0) for free") )
    , l_topology(initLink("topology", "link to the topology container"))
    , finished(false)
{
    // default to indice 0
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();

    //default valueEvent to 0
    d_keyTimes.beginEdit()->push_back(0.0 );
    d_keyTimes.endEdit();
    d_keyMovements.beginEdit()->push_back(Deriv() );
    d_keyMovements.endEdit();
    VecBool movedDirection;
    for( unsigned i=0; i<NumDimensions; i++)
        movedDirection[i] = true;
    d_movedDirections.setValue(movedDirection);

    m_indices.setOriginalData(&d_indices);
    m_keyTimes.setOriginalData(&d_keyTimes);
    m_keyMovements.setOriginalData(&d_keyMovements);
    showMovement.setOriginalData(&d_showMovement);
    linearMovementBetweenNodesInIndices.setOriginalData(&d_linearMovementBetweenNodesInIndices);
    mainIndice.setOriginalData(&d_mainIndice);
    minDepIndice.setOriginalData(&d_minDepIndice);
    maxDepIndice.setOriginalData(&d_maxDepIndice);
    m_imposedDisplacmentOnMacroNodes.setOriginalData(&d_imposedDisplacmentOnMacroNodes);
    X0.setOriginalData(&d_X0);
    Y0.setOriginalData(&d_Y0);
    Z0.setOriginalData(&d_Z0);
    movedDirections.setOriginalData(&d_movedDirections);

}


template <class DataTypes>
PartialLinearMovementProjectiveConstraint<DataTypes>::~PartialLinearMovementProjectiveConstraint()
{

}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::clearIndices()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::addIndex(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::removeIndex(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(), index);
    d_indices.endEdit();
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::clearKeyMovements()
{
    d_keyTimes.beginEdit()->clear();
    d_keyTimes.endEdit();
    d_keyMovements.beginEdit()->clear();
    d_keyMovements.endEdit();
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::addKeyMovement(Real time, Deriv movement)
{
    d_keyTimes.beginEdit()->push_back(time );
    d_keyTimes.endEdit();
    d_keyMovements.beginEdit()->push_back(movement );
    d_keyMovements.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    x0.resize(0);
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::reset()
{
    nextT = prevT = 0.0;
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class DataTypes>
template <class DataDeriv>
void PartialLinearMovementProjectiveConstraint<DataTypes>::projectResponseT(DataDeriv& dx,
    const std::function<void(DataDeriv&, const unsigned int, const VecBool&)>& clear)
{
    Real cT = (Real) this->getContext()->getTime();
    VecBool movedDirection = d_movedDirections.getValue();
    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    if (finished && nextT != prevT)
    {
        const SetIndexArray & indices = d_indices.getValue();

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            clear(dx, *it, movedDirection);
        }
    }
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(), [](VecDeriv& dx, const unsigned int index, const VecBool& b)
                               { for (unsigned j = 0; j < b.size(); j++) if (b[j]) dx[index][j] = 0.0; });
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real cT = (Real) this->getContext()->getTime();
    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    if (finished && nextT != prevT)
    {
        const SetIndexArray & indices = d_indices.getValue();

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it] = (nextM - prevM)*(1.0 / (nextT - prevT));
        }
    }
}


template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real cT = (Real) this->getContext()->getTime();

    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = d_indices.getValue();
        x0.resize(x.size());
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x0[*it] = x[*it];
        }
    }

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
    if(finished && nextT != prevT)
    {
        interpolatePosition<Coord>(cT, x.wref());
    }
}

template <class DataTypes>
template <class MyCoord>
void PartialLinearMovementProjectiveConstraint<DataTypes>::interpolatePosition(Real cT, typename std::enable_if<!std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x)
{
    const SetIndexArray & indices = d_indices.getValue();
    Real dt = (cT - prevT) / (nextT - prevT);
    Deriv m = prevM + (nextM-prevM)*dt;
    VecBool movedDirection = d_movedDirections.getValue();
    //set the motion to the Dofs
    if(d_linearMovementBetweenNodesInIndices.getValue())
    {

        const type::vector<Real> &imposedDisplacmentOnMacroNodes = this->d_imposedDisplacmentOnMacroNodes.getValue();
        Real a = d_X0.getValue();
        Real b = d_Y0.getValue();
        Real c = d_Z0.getValue();
        bool case2d=false;
        if((a==0.0)||(b==0.0)||(c==0.0)) case2d=true;
        if(a==0.0) {a=b; b=c;}
        if(b==0.0) {b=c;}

        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for( unsigned j=0; j< NumDimensions; j++)
            {
                if(movedDirection[j])
                {
                    if(case2d)
                    {
                        x[*it][j] = x0[*it][j] + ((Real)1.0/(a*b))*((a-x0[*it][0])*(b-x0[*it][1])*imposedDisplacmentOnMacroNodes[0]+   ///< N1
                                x0[*it][0]*(b-x0[*it][1])*imposedDisplacmentOnMacroNodes[1]+         ///< N2
                                x0[*it][0]*x0[*it][1]*imposedDisplacmentOnMacroNodes[2]+              ///< N3
                                (a-x0[*it][0])*x0[*it][1]*imposedDisplacmentOnMacroNodes[3])*m[j];    ///< N4
                        //                             4|----------------|3
                        //                              |                |
                        //                              |                |
                        //                              |                |
                        //                             1|----------------|2
                    }
                    else ///< case3d
                    {
                        //        |Y
                        // 	      5---------8
                        //       /|	       /|
                        //      / |	      / |
                        //     6--|------7  |
                        //     |  |/	 |  |
                        //     |  1------|--4--->X
                        //     | / 	     | /
                        //     |/	     |/
                        //     2---------3
                        //   Z/
                        //

                        x[*it][j] = x0[*it][j] + ((Real)1.0/(a*b*c))*(
                                    (a-x0[*it][0])*(b-x0[*it][1])*(c-x0[*it][2])*imposedDisplacmentOnMacroNodes[0]+    ///< N1
                                (a-x0[*it][0])*(b-x0[*it][1])*x0[*it][2]*imposedDisplacmentOnMacroNodes[1]+        ///< N2
                                x0[*it][0]*(b-x0[*it][1])*x0[*it][2]*imposedDisplacmentOnMacroNodes[2]+            ///< N3
                                x0[*it][0]*(b-x0[*it][1])*(c-x0[*it][2])*imposedDisplacmentOnMacroNodes[3]+        ///< N4
                                (a-x0[*it][0])*x0[*it][1]*(c-x0[*it][2])*imposedDisplacmentOnMacroNodes[4]+        ///< N5
                                (a-x0[*it][0])*x0[*it][1]*x0[*it][2]*imposedDisplacmentOnMacroNodes[5]+            ///< N6
                                x0[*it][0]*x0[*it][1]*x0[*it][2]*imposedDisplacmentOnMacroNodes[6]+                ///< N7
                                x0[*it][0]*x0[*it][1]*(c-x0[*it][2])*imposedDisplacmentOnMacroNodes[7]             ///< N8

                                )*m[j];

                    }
                }
            }
        }

    }
    else
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for( unsigned j=0; j< NumDimensions; j++)
                if(movedDirection[j]) x[*it][j] = x0[*it][j] + m[j] ;
        }
    }

}

template <class DataTypes>
template <class MyCoord>
void PartialLinearMovementProjectiveConstraint<DataTypes>::interpolatePosition(Real cT, typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x)
{
    const SetIndexArray & indices = d_indices.getValue();

    Real dt = (cT - prevT) / (nextT - prevT);
    Deriv m = prevM + (nextM-prevM)*dt;
    type::Quat<Real> prevOrientation = type::Quat<Real>::createQuaterFromEuler(getVOrientation(prevM));
    type::Quat<Real> nextOrientation = type::Quat<Real>::createQuaterFromEuler(getVOrientation(nextM));

    //set the motion to the Dofs
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        x[*it].getCenter() = x0[*it].getCenter() + getVCenter(m) ;
        x[*it].getOrientation() = x0[*it].getOrientation() * prevOrientation.slerp2(nextOrientation, dt);
    }
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataMatrixDeriv> c = cData;
    projectResponseT<MatrixDeriv>(c.wref(),
        [](MatrixDeriv& res, const unsigned int index, const VecBool& btype)
        {
            auto itRow = res.begin();
            auto itRowEnd = res.end();

            while (itRow != itRowEnd)
            {
                for (auto colIt = itRow.begin(); colIt != itRow.end(); colIt++)
                {
                    if (index == (unsigned int)colIt.index())
                    {
                        Deriv b = colIt.val();
                        for (unsigned int j = 0; j < btype.size(); j++) if (btype[j]) b[j] = 0.0;
                        res.writeLine(itRow.index()).setCol(colIt.index(), b);
                    }
                }
            }
        });
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::findKeyTimes()
{
    Real cT = (Real) this->getContext()->getTime();
    finished = false;

    if(d_keyTimes.getValue().size() != 0 && cT >= *d_keyTimes.getValue().begin() && cT <= *d_keyTimes.getValue().rbegin())
    {
        nextT = *d_keyTimes.getValue().begin();
        prevT = nextT;

        typename type::vector<Real>::const_iterator it_t = d_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_m = d_keyMovements.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while(it_t != d_keyTimes.getValue().end() && !finished)
        {
            if( *it_t <= cT)
            {
                prevT = *it_t;
                prevM = *it_m;
            }
            else
            {
                nextT = *it_t;
                nextM = *it_m;
                finished = true;
            }
            ++it_t;
            ++it_m;
        }
    }
}

// Matrix Integration interface
template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    const SetIndexArray & indices = d_indices.getValue();

    if (core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get()))
    {
        VecBool movedDirection = d_movedDirections.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<NumDimensions; ++c)
            {
                if( movedDirection[c] ) r.matrix->clearRowCol(r.offset + NumDimensions * (*it) + c);
            }
            // Set Fixed Vertex
            for (unsigned int c=0; c<NumDimensions; ++c)
            {
                if( movedDirection[c] ) r.matrix->set(r.offset + NumDimensions * (*it) + c, r.offset + NumDimensions * (*it) + c, 1.0);
            }
        }
    }
}

template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    const int o = matrix->getGlobalOffset(this->mstate.get());
    if (o >= 0) {
        unsigned int offset = (unsigned int)o;
        VecBool movedDirection = d_movedDirections.getValue();
        const SetIndexArray & indices = d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for (unsigned int c = 0; c < NumDimensions; ++c)
            {
                if (movedDirection[c])
                {
                    vector->clear(offset + NumDimensions * (*it) + c);
                }
            }
        }
    }
}


template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix)
{
    static constexpr unsigned int N = Deriv::size();
    const SetIndexArray& indices = d_indices.getValue();
    const VecBool& movedDirection = d_movedDirections.getValue();

    for (const auto index : indices)
    {
        for (unsigned int c = 0; c < N; ++c)
        {
            if (movedDirection[c]) {
                matrix->discardRowCol(N * index + c, N * index + c);
            }
        }
    }
}

//display the path the constrained dofs will go through
template <class DataTypes>
void PartialLinearMovementProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (!vparams->displayFlags().getShowBehaviorModels() || d_keyTimes.getValue().size() == 0)
        return;

    sofa::type::vector<type::Vec3> vertices;
    constexpr sofa::type::RGBAColor color(1, 0.5, 0.5, 1);

    if (d_showMovement.getValue())
    {
        vparams->drawTool()->disableLighting();

        const SetIndexArray & indices = d_indices.getValue();
        const VecDeriv& keyMovements = d_keyMovements.getValue();
        for (unsigned int i = 0; i < keyMovements.size() - 1; i++)
        {
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                const type::Vec3 v0 { DataTypes::getCPos(x0[*it]) + DataTypes::getDPos(keyMovements[i]) };
                const type::Vec3 v1 { DataTypes::getCPos(x0[*it]) + DataTypes::getDPos(keyMovements[i + 1]) };

                vertices.push_back(v0);
                vertices.push_back(v1);
            }
        }
        vparams->drawTool()->drawLines(vertices, 1, color);
    }
    else
    {
        const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

        type::Vec3 point;
        const SetIndexArray & indices = d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            point = DataTypes::getCPos(x[*it]);
            vertices.push_back(point);
        }
        vparams->drawTool()->drawPoints(vertices, 10, color);
    }


}
} // namespace sofa::component::constraint::projective
