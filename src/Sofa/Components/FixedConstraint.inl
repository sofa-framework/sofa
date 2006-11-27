#ifndef SOFA_COMPONENTS_FIXEDCONSTRAINT_INL
#define SOFA_COMPONENTS_FIXEDCONSTRAINT_INL

#include "Sofa/Core/Constraint.inl"
#include "FixedConstraint.h"
#include "GL/template.h"
#include "Common/RigidTypes.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint()
    : Core::Constraint<DataTypes>(NULL)
    , f_indices( dataField(&f_indices,"indices","Indices of the fixed points") )
{}


template <class DataTypes>
FixedConstraint<DataTypes>::FixedConstraint(Core::MechanicalModel<DataTypes>
        * mmodel)
    : Core::Constraint<DataTypes>(mmodel)
    , f_indices( dataField(&f_indices,"indices","Indices of the fixed points") )
{}

template <class DataTypes>
FixedConstraint<DataTypes>::~FixedConstraint()
{}

template <class DataTypes>
FixedConstraint<DataTypes>*  FixedConstraint<DataTypes>::addConstraint(int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
    return this;
}

template <class DataTypes>
FixedConstraint<DataTypes>*  FixedConstraint<DataTypes>::removeConstraint(int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
    return this;
}

// -- Constraint interface


template <class DataTypes>
void FixedConstraint<DataTypes>::init()
{
    this->Core::Constraint<DataTypes>::init();
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
    const SetIndex& indices = f_indices.getValue();
    for (SetIndex::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        res[*it] = Deriv();
    }
}

// Matrix Integration interface
template <class DataTypes>
void FixedConstraint<DataTypes>::applyConstraint(Components::Common::SofaBaseMatrix *mat, unsigned int &offset)
{
    std::cout << "applyConstraint in Matrix with offset = " << offset << std::endl;

    const SetIndex& indices = f_indices.getValue();
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
void FixedConstraint<DataTypes>::applyConstraint(Components::Common::SofaBaseVector *vect, unsigned int &offset)
{
    std::cout << "applyConstraint in Vector with offset = " << offset << std::endl;

    const SetIndex& indices = f_indices.getValue();
    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        vect->element(3 * (*it)
                + offset) = 0.0;
        vect->element(3 * (*it) + offset + 1) = 0.0;
        vect->element(3 * (*it) + offset + 2) = 0.0;
    }
}

template <class DataTypes>
void FixedConstraint<DataTypes>::draw()
{
    if (!getContext()->
        getShowBehaviorModels()) return;
    const VecCoord& x = *this->mmodel->getX();
    //std::cerr<<"FixedConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<endl;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    const SetIndex& indices = f_indices.getValue();
    //std::cerr<<"FixedConstraint<DataTypes>::draw(), indices = "<<indices<<endl;
    for (SetIndex::const_iterator it = indices.begin();
            it != indices.end();
            ++it)
    {
        GL::glVertexT(x[*it]);
    }
    glEnd();
}

// Specialization for rigids
template <>
void FixedConstraint<RigidTypes >::draw();
template <>
void FixedConstraint<RigidTypes >::projectResponse(VecDeriv& dx);

} // namespace Components

} // namespace Sofa

#endif


