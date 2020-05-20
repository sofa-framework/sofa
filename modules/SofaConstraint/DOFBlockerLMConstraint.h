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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_DOFBLOCKERLMCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_DOFBLOCKERLMCONSTRAINT_H
#include "config.h"

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/LMConstraint.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <sofa/simulation/Node.h>


namespace sofa
{

namespace component
{

namespace constraintset
{


/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class DOFBlockerLMConstraintInternalData
{
};




/** Keep two particules at an initial distance
*/
template <class DataTypes>
class DOFBlockerLMConstraint :  public core::behavior::LMConstraint<DataTypes,DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DOFBlockerLMConstraint,DataTypes),SOFA_TEMPLATE2(sofa::core::behavior::LMConstraint, DataTypes, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;


    typedef sofa::component::topology::PointSubsetData< helper::vector<unsigned int> > SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;

    typedef core::ConstraintParams::ConstOrder ConstOrder;


protected:
    DOFBlockerLMConstraintInternalData<DataTypes> data;
    friend class DOFBlockerLMConstraintInternalData<DataTypes>;


    DOFBlockerLMConstraint(MechanicalState *dof = nullptr)
        : core::behavior::LMConstraint<DataTypes, DataTypes>(dof, dof)
        , BlockedAxis(core::objectmodel::Base::initData(&BlockedAxis, "rotationAxis", "List of rotation axis to constrain"))
        , factorAxis(core::objectmodel::Base::initData(&factorAxis, "factorAxis", "Factor to apply in order to block only a certain amount of rotation along the axis"))
        , f_indices(core::objectmodel::Base::initData(&f_indices, "indices", "List of the index of particles to be fixed"))
        , showSizeAxis(core::objectmodel::Base::initData(&showSizeAxis, 1.0f, "showSizeAxis", "size of the vector used to display the constrained axis"))
        , l_topology(initLink("topology", "link to the topology container"))
        , m_pointHandler(nullptr)
    {
        
    }

    ~DOFBlockerLMConstraint()
    {
        if (m_pointHandler)
            delete m_pointHandler;
    }

public:
    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);


    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void resetConstraint() override;

    // -- LMConstraint interface
    void buildConstraintMatrix(const core::ConstraintParams* cParams, core::MultiMatrixDerivId cId, unsigned int &cIndex) override;
    void writeConstraintEquations(unsigned int& lineNumber, core::MultiVecId id, ConstOrder order) override;

    bool isCorrectionComputedWithSimulatedDOF(ConstOrder /*order*/) const override
    {
        simulation::Node* node=(simulation::Node*) this->constrainedObject1->getContext();
        if (node->mechanicalMapping.empty()) return true;
        else return false;
    }

    Data<helper::vector<Deriv> > BlockedAxis; ///< List of rotation axis to constrain
    Data<helper::vector<SReal> > factorAxis; ///< Factor to apply in order to block only a certain amount of rotation along the axis
    SetIndex f_indices; ///< List of the index of particles to be fixed
    Data<float> showSizeAxis; ///< size of the vector used to display the constrained axis

    /// Link to be set to the topology container in the component graph.
    SingleLink<DOFBlockerLMConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    class FCTPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, helper::vector<unsigned int> >
    {
    public:
        FCTPointHandler(DOFBlockerLMConstraint<DataTypes>* _fc, sofa::component::topology::PointSubsetData<helper::vector<unsigned int> >* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, sofa::helper::vector<unsigned int> >(_data), fc(_fc) {}



        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        DOFBlockerLMConstraint<DataTypes> *fc;
    };

protected :
    sofa::helper::vector<SetIndexArray> idxEquations;
    
    FCTPointHandler* m_pointHandler;
};

#if  !defined(SOFA_COMPONENT_CONSTRAINTSET_DOFBLOCKERLMCONSTRAINT_CPP)
extern template class DOFBlockerLMConstraint<defaulttype::Rigid3Types>;
extern template class DOFBlockerLMConstraint<defaulttype::Vec3Types>;

#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
