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
#ifndef SOFA_COMPONENT_CONSTRAINT_LINEARMOVEMENTCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_LINEARMOVEMENTCONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/component/constraint/LinearMovementConstraint.h>
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
bool LinearMovementConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    LinearMovementConstraint<DataTypes> *fc= (LinearMovementConstraint<DataTypes> *)param;
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
void LinearMovementConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    LinearMovementConstraint<DataTypes> *fc= (LinearMovementConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeIndice((unsigned int) pointIndex);
    }
    return;
}

template <class DataTypes>
LinearMovementConstraint<DataTypes>::LinearMovementConstraint()
    : core::componentmodel::behavior::Constraint<DataTypes>(NULL)
    , f_indices( initData(&f_indices,"indices","Indices of the constrained points") )
    , f_keyTimes(  initData(&f_keyTimes,"keyTimes","key times for the movements") )
    , f_keyMovements(  initData(&f_keyMovements,"movements","movements corresponding to the key times") )
{
    // default to indice 0
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();

    //default valueEvent to 0
    f_keyTimes.beginEdit()->push_back( 0.0 );
    f_keyTimes.endEdit();
    f_keyMovements.beginEdit()->push_back( Deriv() );
    f_keyMovements.endEdit();
}


// Handle topological changes
template <class DataTypes> void LinearMovementConstraint<DataTypes>::handleTopologyChange()
{
    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    f_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());

}

template <class DataTypes>
LinearMovementConstraint<DataTypes>::~LinearMovementConstraint()
{
}

template <class DataTypes>
void LinearMovementConstraint<DataTypes>::clearIndices()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void LinearMovementConstraint<DataTypes>::addIndice(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void LinearMovementConstraint<DataTypes>::removeIndice(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}

template <class DataTypes>
void LinearMovementConstraint<DataTypes>::clearTranslations()
{
    f_keyTimes.beginEdit()->clear();
    f_keyTimes.endEdit();
    f_keyMovements.beginEdit()->clear();
    f_keyMovements.endEdit();
}

template <class DataTypes>
void LinearMovementConstraint<DataTypes>::addTranslation(Real time, Deriv movement)
{
    f_keyTimes.beginEdit()->push_back( time );
    f_keyTimes.endEdit();
    f_keyMovements.beginEdit()->push_back( movement );
    f_keyMovements.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void LinearMovementConstraint<DataTypes>::init()
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
void LinearMovementConstraint<DataTypes>::projectResponse(VecDeriv& dx)
{
    const SetIndexArray & indices = f_indices.getValue().getArray();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        dx[*it] = Deriv();
    }
}


template <class DataTypes>
void LinearMovementConstraint<DataTypes>::projectPosition(VecCoord& x)
{
    const SetIndexArray & indices = f_indices.getValue().getArray();

    Deriv m1, m2;
    double cT = this->getContext()->getTime();
    bool finished=false;

    if(cT==0.0)
    {
        prevMovement = Deriv();
    }


    typename helper::vector<Real>::const_iterator it_t = f_keyTimes.getValue().begin();
    typename VecDeriv::const_iterator it_m = f_keyMovements.getValue().begin();

    //WARNING : we consider that the key-events are in chronological order
    //here we search between which keyTimes we are, to know which are the movements to interpolate
    while( it_t != f_keyTimes.getValue().end() && !finished)
    {
        if( *it_t <= cT)
        {
            prevT = *it_t;
            if(prevT == nextT)
            {
                prevMovement = Deriv();
            }
            m1 = *it_m;
        }
        else
        {
            nextT = *it_t;
            m2 = *it_m;
            finished = true;
        }
        it_t++;
        it_m++;
    }

    //if we found 2 keyTimes, we have to interpolate a movement
    if(finished)
    {
        double dt = (cT - prevT) / (nextT - prevT);
        Deriv m= (m2-m1)*dt;

        m -= prevMovement;
        prevMovement += m;

        //set the movment to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it] += m;
        }
    }

}

// Matrix Integration interface
/*template <class DataTypes>
void LinearMovementConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int &offset)
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
void LinearMovementConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int &offset)
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
*/

template <class DataTypes>
void LinearMovementConstraint<DataTypes>::draw()
{
    if (!getContext()->
        getShowBehaviorModels()) return;
    const VecCoord& x = *this->mstate->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    const SetIndexArray & indices = f_indices.getValue().getArray();
    for (SetIndexArray::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        gl::glVertexT(x[*it]);
    }
    glEnd();
}

// Specialization for rigids
template <>
void LinearMovementConstraint<Rigid3dTypes >::draw();
template <>
void LinearMovementConstraint<Rigid3fTypes >::draw();

} // namespace constraint

} // namespace component

} // namespace sofa

#endif


