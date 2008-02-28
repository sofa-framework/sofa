/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINT_ATTACHCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_ATTACHCONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/PairInteractionConstraint.inl>
#include <sofa/component/constraint/AttachConstraint.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

#include <sofa/component/topology/PointSubset.h>

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace core::componentmodel::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::componentmodel::behavior;

#if 0
// Define TestNewPointFunction
template< class DataTypes>
bool AttachConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    AttachConstraint<DataTypes> *fc= (AttachConstraint<DataTypes> *)param;
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
void AttachConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    AttachConstraint<DataTypes> *fc= (AttachConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}
#endif

template <class DataTypes>
AttachConstraint<DataTypes>::AttachConstraint()
    : f_indices1( initData(&f_indices1,"indices1","Indices of the source points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the fixed points on the second model") )
    , f_radius( initData(&f_radius,(Real)-1,"radius", "Radius to search corresponding fixed point if no indices are given") )
    , f_twoWay( initData(&f_twoWay,false,"twoWay", "true if forces should be projected back from model2 to model1") )
{
    // default to indice 0
//     f_indices1.beginEdit()->push_back(0);
//     f_indices1.endEdit();
//     f_indices2.beginEdit()->push_back(0);
//     f_indices2.endEdit();
}

#if 0
// Handle topological changes
template <class DataTypes> void AttachConstraint<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
}
#endif

template <class DataTypes>
AttachConstraint<DataTypes>::~AttachConstraint()
{
}

template <class DataTypes>
void AttachConstraint<DataTypes>::clearConstraints()
{
    f_indices1.beginEdit()->clear();
    f_indices1.endEdit();
    f_indices2.beginEdit()->clear();
    f_indices2.endEdit();
}

template <class DataTypes>
void AttachConstraint<DataTypes>::addConstraint(unsigned int index1, unsigned int index2)
{
    f_indices1.beginEdit()->push_back(index1);
    f_indices1.endEdit();
    f_indices2.beginEdit()->push_back(index2);
    f_indices2.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void AttachConstraint<DataTypes>::init()
{
    this->core::componentmodel::behavior::PairInteractionConstraint<DataTypes>::init();
    if (f_radius.getValue() >= 0 && f_indices1.getValue().size()==0 && f_indices2.getValue().size()==0 && this->mstate1 && this->mstate2)
    {
        const Real maxR = f_radius.getValue();
        VecCoord& x1 = *this->mstate1->getX();
        VecCoord& x2 = *this->mstate2->getX();
        for (unsigned int i2=0; i2<x2.size(); ++i2)
        {
            int best = -1;
            Real bestR = maxR;
            for (unsigned int i1=0; i1<x1.size(); ++i1)
            {
                Real r = (x2[i2]-x1[i1]).norm();
                if (r <= bestR)
                {
                    best = i1;
                    bestR = r;
                }
            }
            if (best >= 0)
            {
                addConstraint(best, i2);
            }
        }
    }
#if 0
    // Initialize functions and parameters
    topology::PointSubset my_subset = f_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );
#endif
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectPosition(VecCoord& res1, VecCoord& res2)
{
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        res2[indices2[i]] = res1[indices1[i]];
    }
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectVelocity(VecDeriv& res1, VecDeriv& res2)
{
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        res2[indices2[i]] = res1[indices1[i]];
    }
}

template <class DataTypes>
void AttachConstraint<DataTypes>::projectResponse(VecDeriv& res1, VecDeriv& res2)
{
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    bool twoway = f_twoWay.getValue();
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        if (twoway)
        {
            res1[indices1[i]] += res2[indices2[i]];
            res2[indices2[i]] = res1[indices1[i]];
        }
        else
        {
            res2[indices2[i]] = Deriv();
        }
    }
}

// Matrix Integration interface
template <class DataTypes>
void AttachConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int &offset)
{
    std::cout << "applyConstraint in Matrix with offset = " << offset << std::endl;
    const SetIndexArray & indices = f_indices2.getValue().getArray();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        // Reset Attach Row
        for (int i=0; i<mat->colSize(); i++)
        {
            mat->element(i, 3 * (*it) + offset) = 0.0;
            mat->element(i, 3 * (*it) + offset + 1) = 0.0;
            mat->element(i, 3 * (*it) + offset + 2) = 0.0;
        }

        // Reset Attach Col
        for (int i=0; i<mat->rowSize(); i++)
        {
            mat->element(3 * (*it) + offset, i) = 0.0;
            mat->element(3 * (*it) + offset + 1, i) = 0.0;
            mat->element(3 * (*it) + offset + 2, i) = 0.0;
        }

        // Set Attach Vertex
        mat->element(3 * (*it) + offset, 3 * (*it) + offset) = 1.0;
        mat->element(3 * (*it) + offset, 3 * (*it) + offset + 1) = 0.0;
        mat->element(3 * (*it) + offset, 3 * (*it) + offset + 2) = 0.0;

        mat->element(3 * (*it) + offset + 1, 3 * (*it) + offset) = 0.0;
        mat->element(3 * (*it) + offset + 1, 3 * (*it) + offset + 1) = 1.0;
        mat->element(3 * (*it) + offset + 1, 3 * (*it) + offset + 2) = 0.0;

        mat->element(3 * (*it) + offset + 2, 3 * (*it) + offset) = 0.0;
        mat->element(3 * (*it) + offset + 2, 3 * (*it) + offset + 1) = 0.0;
        mat->element(3 * (*it) + offset + 2, 3 * (*it) + offset + 2) = 1.0;
    }
}

template <class DataTypes>
void AttachConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int &offset)
{
    std::cout << "applyConstraint in Vector with offset = " << offset << std::endl;

    const SetIndexArray & indices = f_indices2.getValue().getArray();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        vect->element(3 * (*it) + offset) = 0.0;
        vect->element(3 * (*it) + offset + 1) = 0.0;
        vect->element(3 * (*it) + offset + 2) = 0.0;
    }
}


template <class DataTypes>
void AttachConstraint<DataTypes>::draw()
{
    if (!getContext()->
        getShowBehaviorModels()) return;
    const SetIndexArray & indices1 = f_indices1.getValue().getArray();
    const SetIndexArray & indices2 = f_indices2.getValue().getArray();
    VecCoord& x1 = *this->mstate1->getX();
    VecCoord& x2 = *this->mstate2->getX();
    glDisable (GL_LIGHTING);
    glPointSize(5);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        gl::glVertexT(x2[indices2[i]]);
    }
    glEnd();
    glPointSize(1);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    for (unsigned int i=0; i<indices1.size() && i<indices2.size(); ++i)
    {
        gl::glVertexT(x1[indices1[i]]);
        gl::glVertexT(x2[indices2[i]]);
    }
    glEnd();
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
