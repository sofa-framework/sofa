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
#include <sofa/component/constraint/lagrangian/model/SlidingLagrangianConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/constraint/lagrangian/model/BilateralLagrangianConstraint.h>
#include <sofa/component/constraint/lagrangian/model/UnilateralLagrangianConstraint.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/Vec.h>
namespace sofa::component::constraint::lagrangian::model
{

template<class DataTypes>
SlidingLagrangianConstraint<DataTypes>::SlidingLagrangianConstraint()
    : SlidingLagrangianConstraint(nullptr, nullptr)
{
}

template<class DataTypes>
SlidingLagrangianConstraint<DataTypes>::SlidingLagrangianConstraint(MechanicalState* object)
    : SlidingLagrangianConstraint(object, object)
{
}

template<class DataTypes>
SlidingLagrangianConstraint<DataTypes>::SlidingLagrangianConstraint(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
    , d_m1(initData(&d_m1, 0, "sliding_point","index of the spliding point on the first model"))
    , d_m2a(initData(&d_m2a, 0, "axis_1","index of one end of the sliding axis"))
    , d_m2b(initData(&d_m2b, 0, "axis_2","index of the other end of the sliding axis"))
    , d_force(initData(&d_force,"force","force (impulse) used to solve the constraint"))
{
}

template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::init()
{
    assert(this->mstate1);
    assert(this->mstate2);

}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::buildConstraintMatrix(const core::ConstraintParams*, DataMatrixDeriv &c1_d, DataMatrixDeriv &c2_d, unsigned int &cIndex
        , const DataVecCoord &x1, const DataVecCoord &x2)
{

    int tm1 =  d_m1.getValue();
    int tm2a = d_m2a.getValue();
    int tm2b = d_m2b.getValue();

    auto c1 = sofa::helper::getWriteAccessor(c1_d);
    auto c2 = sofa::helper::getWriteAccessor(c2_d);

    const Coord_t<DataTypes> P = x1.getValue()[tm1];
    const Coord_t<DataTypes> A = x2.getValue()[tm2a];
    const Coord_t<DataTypes> B = x2.getValue()[tm2b];

    // the axis
    Deriv_t<DataTypes> dirAxe;
    DataTypes::setDPos(dirAxe, DataTypes::getCPos(B) - DataTypes::getCPos(A));
    const Real ab = dirAxe.norm();

    m_constraintDirections.clear();

    m_projectionBarycentricCoordinate=0; //Coef to point B -> proj = bary*B + (1-bary)*A

    if ( ab < std::numeric_limits<Real>::epsilon() ) // If A and B are at the same position, then a full dof constraint must be applied totally linking position of P and A
    {
        for (unsigned i=0; i<Deriv_t<DataTypes>::spatial_dimensions; ++i)
        {
            Deriv_t<DataTypes> temp;
            temp[i] = 1;
            m_constraintDirections.push_back(temp);
        }
    }
    else
    {
        // Normalize direction
        DataTypes::setDPos(dirAxe,DataTypes::getDPos(dirAxe).normalized());

        // Distance to point A on sliding direction
        Real r = dot(DataTypes::getCPos(P) - DataTypes::getCPos(A) , DataTypes::getDPos(dirAxe));
        // Normalized distance to point A on sliding direction, if equal to 1 then it is the same distance to A as B
        Real r2 = r / ab;

        // Compute bary coef of normalized distance (if proj is outside of edge, it is forced to be applied on one side, either A (bary==0) or B (bary==1)
        m_projectionBarycentricCoordinate = r2 < 0 ?  0 : r2;
        m_projectionBarycentricCoordinate = m_projectionBarycentricCoordinate > 1.0 ?  1.0 : m_projectionBarycentricCoordinate;

        // This is the coordinates of the projected point
        Coord_t<DataTypes> proj;
        DataTypes::setCPos(proj, DataTypes::getCPos(A) + DataTypes::getDPos(dirAxe) * r);

        // Compute projection direction
        Deriv_t<DataTypes> dirProj;
        DataTypes::setDPos(dirProj, DataTypes::getCPos(P) -  DataTypes::getCPos(proj));
        // If the projection is too close to the real point, create a proj dir that is randomly chosen around the edge
        if (DataTypes::getDPos(dirProj).norm() < std::numeric_limits<Real>::epsilon())
        {
            typename DataTypes::DPos xVec;
            xVec[0] = 1;
            if ( cross(xVec, DataTypes::getDPos(dirAxe)).norm() < std::numeric_limits<Real>::epsilon())
                xVec[1] = 1;

            DataTypes::setDPos(dirProj, cross(xVec, DataTypes::getDPos(dirAxe)));
        }
        DataTypes::setDPos(dirProj,DataTypes::getDPos(dirProj).normalized()); // direction of the constraint

        // Compute second normal that complete the set of constraint required to pull the point on the edge (only when dimension is supérior to 2)
        Deriv_t<DataTypes> dirOrtho;
        if constexpr ( Deriv_t<DataTypes>::spatial_dimensions > 2 )
        {
            DataTypes::setDPos(dirOrtho, cross(DataTypes::getDPos(dirProj), DataTypes::getDPos(dirAxe)).normalized());
        }

        m_constraintDirections.push_back(dirProj);
        if constexpr ( Deriv_t<DataTypes>::spatial_dimensions > 2 )
        {
            m_constraintDirections.push_back(dirOrtho);
        }
        m_constraintDirections.push_back(dirAxe);
        m_constraintDirections.push_back(-dirAxe);
    }

    //Now add vectors to the constraint matrix
    if(m_constraintDirections.size()==Deriv_t<DataTypes>::spatial_dimensions)
    {
        //When A=B we don't care what point ton constraint, we just ocnstraint A
        for(const auto & dirVec : m_constraintDirections)
        {
            auto c1_it = c1->writeLine(cIndex);
            c1_it.setCol(tm1, dirVec);

            auto c2_it = c2->writeLine(cIndex);
            c2_it.addCol(tm2a, -dirVec);

            ++cIndex;
        }
    }
    else
    {
        //When A!=B we want to allow motion between A and B, we must be careful on the constraint
        //direction of the constraint applied to A and B so the Unilateral constraints allow motion
        //inbetween the two points  A--------------B
        //               (forbidden)|->  (free)   <-|(forbidden)

        for(unsigned i=0; i<m_constraintDirections.size()  - 2 ; ++i)
        {
            auto c1_it = c1->writeLine(cIndex);
            c1_it.addCol(tm1, m_constraintDirections[i]);

            auto c2_it = c2->writeLine(cIndex);
            c2_it.addCol(tm2a, -m_constraintDirections[i] * (1-m_projectionBarycentricCoordinate));
            c2_it.addCol(tm2b, -m_constraintDirections[i] * m_projectionBarycentricCoordinate);

            ++cIndex;
        }

        auto c1_it_a = c1->writeLine(cIndex);
        c1_it_a.setCol(tm1, m_constraintDirections[ m_constraintDirections.size() -2 ]);
        auto c1_it_b = c1->writeLine(cIndex + 1);
        c1_it_b.setCol(tm1, m_constraintDirections[ m_constraintDirections.size() -1 ]);


        auto c2_it_a = c2->writeLine(cIndex);
        c2_it_a.addCol(tm2a, -m_constraintDirections[ m_constraintDirections.size() -2 ]);
        auto c2_it_b = c2->writeLine(cIndex + 1);
        c2_it_b.addCol(tm2b, -m_constraintDirections[ m_constraintDirections.size() -1 ]);

        cIndex += 2;
    }
}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::getConstraintViolation(const core::ConstraintParams *, linearalgebra::BaseVector *v, const DataVecCoord & x1, const DataVecCoord & x2
        , const DataVecDeriv &, const DataVecDeriv &)
{
    const Coord_t<DataTypes> P = x1.getValue()[d_m1.getValue()];
    const Coord_t<DataTypes> A = x2.getValue()[d_m2a.getValue()];
    const Coord_t<DataTypes> B = x2.getValue()[d_m2b.getValue()];

    const auto constraintIndex = this->d_constraintIndex.getValue();

    if(m_constraintDirections.size()==Deriv_t<DataTypes>::spatial_dimensions) // A and B are on the same point
    {
        //If A=B, we have bilateral constraint that fixed all dofs, they are thus aligned with the world system of coordinates and the violation is easily computed
        typename DataTypes::CPos temp = DataTypes::getCPos(P)-DataTypes::getCPos(A);
        for(unsigned i=0; i<m_constraintDirections.size() ; ++i)
        {
            v->set(constraintIndex + i, temp[i] );
        }
    }
    else
    {
        typename DataTypes::CPos newProj = DataTypes::getCPos(B)*m_projectionBarycentricCoordinate + DataTypes::getCPos(A)*(1-m_projectionBarycentricCoordinate);
        typename DataTypes::DPos PtoProj = DataTypes::getCPos(P) - newProj;
        typename DataTypes::DPos PtoA = DataTypes::getCPos(P) - DataTypes::getCPos(A);
        typename DataTypes::DPos PtoB = DataTypes::getCPos(P) - DataTypes::getCPos(B);

        for(unsigned i=0; i<m_constraintDirections.size()  - 2 ; ++i)
        {
            v->set(constraintIndex + i, dot(PtoProj, DataTypes::getDPos(m_constraintDirections[i])) );
        }
        v->set(constraintIndex + 2, dot(PtoA ,  DataTypes::getDPos(m_constraintDirections[2])) );
        v->set(constraintIndex + 3, dot(PtoB ,  DataTypes::getDPos(m_constraintDirections[3])) );
    }
}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::getConstraintResolution(const ConstraintParams*,
                                                           std::vector<core::behavior::ConstraintResolution*>& resTab,
                                                           unsigned int& offset)
{
    if(m_constraintDirections.size()==Deriv_t<DataTypes>::spatial_dimensions)
    {
        //If A=B only bilateral constraints
        for(unsigned i=0; i<m_constraintDirections.size() ; ++i)
        {
            resTab[offset++] = new BilateralConstraintResolution();
        }
    }
    else
    {
        // If A!=B, constraints applied to A and B (along sliding direction) must be unilateral
        for(unsigned i=0; i<m_constraintDirections.size()  - 2 ; ++i)
        {
            resTab[offset++] = new BilateralConstraintResolution();
        }
        resTab[offset++] = new UnilateralConstraintResolution();
        resTab[offset++] = new UnilateralConstraintResolution();
    }

}


template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::storeLambda(const ConstraintParams* /*cParams*/, sofa::core::MultiVecDerivId /*res*/, const sofa::linearalgebra::BaseVector* lambda)
{
    Deriv_t<DataTypes> force;
    const auto constraintIndex = this->d_constraintIndex.getValue();
    for (unsigned i=0; i< m_constraintDirections.size(); ++i )
    {
        force += m_constraintDirections[i]*lambda->element(constraintIndex+i);
    }
    d_force.setValue(force);
}

template<class DataTypes>
void SlidingLagrangianConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields())
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    vparams->drawTool()->disableLighting();

    sofa::type::RGBAColor color;

    if(m_constraintDirections.size() == Deriv_t<DataTypes>::spatial_dimensions)
        color = sofa::type::RGBAColor::yellow();
    else
        color = sofa::type::RGBAColor::green();


    std::vector<typename DataTypes::CPos> vertices;
    vertices.push_back(DataTypes::getCPos((this->mstate1->read(core::vec_id::read_access::position)->getValue())[d_m1.getValue()]));

    vparams->drawTool()->drawPoints(vertices, 10, color);
    vertices.clear();

    color = sofa::type::RGBAColor::blue();
    vertices.push_back(DataTypes::getCPos((this->mstate2->read(core::vec_id::read_access::position)->getValue())[d_m2a.getValue()]));
    vertices.push_back(DataTypes::getCPos((this->mstate2->read(core::vec_id::read_access::position)->getValue())[d_m2b.getValue()]));
    vparams->drawTool()->drawLines(vertices, 1, color);


}

} //namespace sofa::component::constraint::lagrangian::model
