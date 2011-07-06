/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINTSET_FIXEDLMCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_FIXEDLMCONSTRAINT_H

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/LMConstraint.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/simulation/common/Node.h>


namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::core::topology;
/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class FixedLMConstraintInternalData
{
};




/** Keep two particules at an initial distance
*/
template <class DataTypes>
class FixedLMConstraint :  public core::behavior::LMConstraint<DataTypes,DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedLMConstraint,DataTypes),SOFA_TEMPLATE2(sofa::core::behavior::LMConstraint, DataTypes, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;


    typedef sofa::component::topology::PointSubset SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;

    typedef core::ConstraintParams::ConstOrder ConstOrder;

protected:
    FixedLMConstraintInternalData<DataTypes> data;
    friend class FixedLMConstraintInternalData<DataTypes>;

public:
    FixedLMConstraint( MechanicalState *dof)
        : core::behavior::LMConstraint<DataTypes,DataTypes>(dof,dof)
        , f_indices(core::objectmodel::Base::initData(&f_indices, "indices", "List of the index of particles to be fixed"))
        , _drawSize(core::objectmodel::Base::initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    {};

    FixedLMConstraint()
        : f_indices(core::objectmodel::Base::initData(&f_indices, "indices", "List of the index of particles to be fixed"))
        , _drawSize(core::objectmodel::Base::initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    {};

    ~FixedLMConstraint() {};

    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);

    // Handle topological changes
    virtual void handleTopologyChange();

    void init();
    void draw(const core::visual::VisualParams*);
    void initFixedPosition();
    void reset() {initFixedPosition();};

    // -- LMConstraint interface
    void buildConstraintMatrix(const core::ConstraintParams* cParams /* PARAMS FIRST */, core::MultiMatrixDerivId cId, unsigned int &cIndex);
    void writeConstraintEquations(unsigned int& lineNumber, core::MultiVecId id, ConstOrder order);


    std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const FixedLMConstraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    bool isCorrectionComputedWithSimulatedDOF(ConstOrder /*order*/) const
    {
        simulation::Node* node=(simulation::Node*) this->constrainedObject1->getContext();
        if (node->mechanicalMapping.empty()) return true;
        else return false;
    }

    bool useMask() const {return true;}

    Data<SetIndex> f_indices;
    Data<double> _drawSize;

protected :

    Deriv X,Y,Z;
    SetIndexArray idxX, idxY, idxZ;
    std::map< unsigned int, Coord> restPosition;




    sofa::core::topology::BaseMeshTopology* topology;


    // Define TestNewPointFunction
    static bool FCTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& );

    // Define RemovalFunction
    static void FCRemovalFunction ( int , void*);

};


#if defined(WIN32) && !defined(SOFA_BUILD_COMPONENT_CONSTRAINTSET)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_CONSTRAINTSET_API FixedLMConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API FixedLMConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_CONSTRAINTSET_API FixedLMConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API FixedLMConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
