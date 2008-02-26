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
#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_FIXEDCONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/component/constraint/FixedConstraint.h>
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


// Define TestNewPointFunction
template< class DataTypes>
bool FixedConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    FixedConstraint<DataTypes> *fc= (FixedConstraint<DataTypes> *)param;
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
void FixedConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    FixedConstraint<DataTypes> *fc= (FixedConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
    return;
}

template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint()
    : core::componentmodel::behavior::Constraint<DataTypes>(NULL)
    , f_indices( initData(&f_indices,"indices","Indices of the fixed points") )
    , f_fixAll( initData(&f_fixAll,false,"fixAll","filter all the DOF to implement a fixed object") )
{
    // default to indice 0
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}


// Handle topological changes
template <class DataTypes> void FixedConstraint<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());

}

template <class DataTypes>
FixedConstraint<DataTypes>::~FixedConstraint()
{
}

template <class DataTypes>
void FixedConstraint<DataTypes>::clearConstraints()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void FixedConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void FixedConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void FixedConstraint<DataTypes>::init()
{
    this->core::componentmodel::behavior::Constraint<DataTypes>::init();

    // Initialize functions and parameters
    topology::PointSubset my_subset = f_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );

}

template <class DataTypes>
void FixedConstraint<DataTypes>::projectResponse(VecDeriv& res)
{
    const SetIndexArray & indices = f_indices.getValue().getArray();
    //std::cerr<<"FixedConstraint<DataTypes>::projectResponse, res.size()="<<res.size()<<endl;
    if( f_fixAll.getValue()==true )    // fix everyting
    {
        for( unsigned i=0; i<res.size(); i++ )
            res[i] = Deriv();
    }
    else
    {
        for (SetIndexArray::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            res[*it] = Deriv();
        }
    }
}

// Matrix Integration interface
template <class DataTypes>
void FixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int &offset)
{
    std::cout << "applyConstraint in Matrix with offset = " << offset << std::endl;
    const SetIndexArray & indices = f_indices.getValue().getArray();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        // Reset Fixed Row
        for (int i=0; i<mat->colSize(); i++)
        {
            mat->element(i, 3 * (*it) + offset) = 0.0;
            mat->element(i, 3 * (*it) + offset + 1) = 0.0;
            mat->element(i, 3 * (*it) + offset + 2) = 0.0;
        }

        // Reset Fixed Col
        for (int i=0; i<mat->rowSize(); i++)
        {
            mat->element(3 * (*it) + offset, i) = 0.0;
            mat->element(3 * (*it) + offset + 1, i) = 0.0;
            mat->element(3 * (*it) + offset + 2, i) = 0.0;
        }

        // Set Fixed Vertex
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
void FixedConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int &offset)
{
    std::cout << "applyConstraint in Vector with offset = " << offset << std::endl;

    const SetIndexArray & indices = f_indices.getValue().getArray();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        vect->element(3 * (*it) + offset) = 0.0;
        vect->element(3 * (*it) + offset + 1) = 0.0;
        vect->element(3 * (*it) + offset + 2) = 0.0;
    }
}


template <class DataTypes>
void FixedConstraint<DataTypes>::draw()
{
    if (!getContext()->
        getShowBehaviorModels()) return;
    const VecCoord& x = *this->mstate->getX();
    //std::cerr<<"FixedConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<endl;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    const SetIndexArray & indices = f_indices.getValue().getArray();
    //std::cerr<<"FixedConstraint<DataTypes>::draw(), indices = "<<indices<<endl;
    if( f_fixAll.getValue()==true ) for (unsigned i=0; i<x.size(); i++ )
        {
            gl::glVertexT(x[i]);
        }
    else for (SetIndexArray::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            gl::glVertexT(x[*it]);
        }
    glEnd();
}

// Specialization for rigids
template <>
void FixedConstraint<Rigid3dTypes >::draw();
template <>
void FixedConstraint<Rigid3fTypes >::draw();
template <>
void FixedConstraint<Rigid2dTypes >::draw();
template <>
void FixedConstraint<Rigid2fTypes >::draw();

} // namespace constraint

} // namespace component

} // namespace sofa

#endif


