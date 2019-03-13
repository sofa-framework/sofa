/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_CPP
#include <SofaBoundaryCondition/FixedConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


int FixedConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
        .add< FixedConstraint<Vec3Types> >()
        .add< FixedConstraint<Vec2Types> >()
        .add< FixedConstraint<Vec1Types> >()
        .add< FixedConstraint<Vec6Types> >()
        .add< FixedConstraint<Rigid3Types> >()
        .add< FixedConstraint<Rigid2Types> >()

        ;


// methods specilizations declaration
template <> SOFA_BOUNDARY_CONDITION_API
void FixedConstraint<defaulttype::Rigid3Types >::draw(const core::visual::VisualParams* vparams);
template <> SOFA_BOUNDARY_CONDITION_API
void FixedConstraint<defaulttype::Rigid2Types >::draw(const core::visual::VisualParams* vparams);



template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec3Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec2Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec1Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Vec6Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Rigid3Types>;
template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<Rigid2Types>;



// methods specilizations definition
template <>
void FixedConstraint<Rigid3Types>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!d_showObject.getValue()) return;
    if (!this->isActive()) return;

    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    std::vector< Vector3 > points;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    if( d_fixAll.getValue()==true )
        for (unsigned i=0; i<x.size(); i++ )
            points.push_back(x[i].getCenter());
    else
    {
        if( x.size() < indices.size() )
        {
            for (unsigned i=0; i<x.size(); i++ )
                points.push_back(x[indices[i]].getCenter());
        }
        else
        {
            for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
                points.push_back(x[*it].getCenter());
        }
    }

    if( d_drawSize.getValue() == 0) // old classical drawing by points
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    else
        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
}

template <>
void FixedConstraint<Rigid2Types>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!d_showObject.getValue()) return;
    if (!this->isActive()) return;

    const SetIndexArray & indices = d_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    vparams->drawTool()->saveLastState();

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    vparams->drawTool()->setLightingEnabled(false);
    sofa::defaulttype::Vec4f color (1,0.5,0.5,1);
    std::vector<sofa::defaulttype::Vector3> vertices;

    if( d_fixAll.getValue()==true )
    {
        for (unsigned i=0; i<x.size(); i++ )
            vertices.push_back(sofa::defaulttype::Vector3(x[i].getCenter()[0],
                                                          x[i].getCenter()[1],
                                                          0.0));
    }
    else
    {
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
            vertices.push_back(sofa::defaulttype::Vector3(x[*it].getCenter()[0],
                                                          x[*it].getCenter()[1],
                                                          0.0));
    }

    vparams->drawTool()->drawPoints(vertices, 10, color);
    vparams->drawTool()->restoreLastState();
}


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

