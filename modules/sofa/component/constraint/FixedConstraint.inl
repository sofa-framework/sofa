#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_FIXEDCONSTRAINT_INL

#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/component/constraint/FixedConstraint.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint()
    : core::componentmodel::behavior::Constraint<DataTypes>(NULL)
    , f_indices( dataField(&f_indices,"indices","Indices of the fixed points") )
{}


template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>
        * mstate)
    : core::componentmodel::behavior::Constraint<DataTypes>(mstate)
    , f_indices( dataField(&f_indices,"indices","Indices of the fixed points") )
{}

template <class DataTypes>
FixedConstraint<DataTypes>::~FixedConstraint()
{}

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
    f_listening.setValue(true);
    // sort indices and remove duplicates by copying them to a std::set
    //SetIndex& indices = *f_indices.beginEdit();
    //std::set<int> tmpset(indices.begin(), indices.end());
    //indices = SetIndex(tmpset.begin(),tmpset.end());
    //f_indices.endEdit();
}

template <class DataTypes>
void FixedConstraint<DataTypes>::projectResponse(VecDeriv& res)
{
    //std::cerr<<"FixedConstraint<DataTypes>::projectResponse, res.size()="<<res.size()<<endl;
    const SetIndexArray & indices = f_indices.getValue().getArray();
    for (SetIndex::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        res[*it] = Deriv();
    }
}

// Matrix Integration interface
template <class DataTypes>
void FixedConstraint<DataTypes>::applyConstraint(defaulttype::SofaBaseMatrix *mat, unsigned int &offset)
{
    std::cout << "applyConstraint in Matrix with offset = " << offset << std::endl;
    const SetIndexArray & indices = f_indices.getValue().getArray();

    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        // Reset Fixed Row
        for (int i=0; i<mat->colDim(); i++)
        {
            mat->element(i, 3 * (*it) + offset) = 0.0;
            mat->element(i, 3 * (*it) + offset + 1) = 0.0;
            mat->element(i, 3 * (*it) + offset + 2) = 0.0;
        }

        // Reset Fixed Col
        for (int i=0; i<mat->rowDim(); i++)
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
void FixedConstraint<DataTypes>::applyConstraint(defaulttype::SofaBaseVector *vect, unsigned int &offset)
{
    std::cout << "applyConstraint in Vector with offset = " << offset << std::endl;

    const SetIndexArray & indices = f_indices.getValue().getArray();
    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        vect->element(3 * (*it)
                + offset) = 0.0;
        vect->element(3 * (*it) + offset + 1) = 0.0;
        vect->element(3 * (*it) + offset + 2) = 0.0;
    }
}

template <class DataTypes>
void FixedConstraint<DataTypes>::handleEvent( Event *event )
{
    topology::TopologyChangedEvent *tce=dynamic_cast<topology::TopologyChangedEvent *>(event);
    /// test that the event is a change of topology and that it
    if ((tce) && (tce->getTopology()== getContext()->getMainTopology()))
    {
        core::componentmodel::topology::BaseTopology *topology = static_cast<core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

        std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itBegin=topology->firstChange();
        std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator itEnd=topology->lastChange();
        std::list<const core::componentmodel::topology::TopologyChange *>::const_iterator it;

        SetIndex & indices = *f_indices.beginEdit();
        indices.handleTopologyEvents(itBegin,itEnd);
        f_indices.endEdit();

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
    for (SetIndex::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        gl::glVertexT(x[*it]);
    }
    glEnd();
}

// Specialization for rigids
template <>
void FixedConstraint<RigidTypes >::draw();
template <>
void FixedConstraint<RigidTypes >::projectResponse(VecDeriv& dx);

} // namespace constraint

} // namespace component

} // namespace sofa

#endif


