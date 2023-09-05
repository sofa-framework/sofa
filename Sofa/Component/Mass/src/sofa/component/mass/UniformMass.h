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
#include <sofa/component/mass/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

#include <sofa/component/mass/VecMassType.h>
#include <sofa/component/mass/RigidMassType.h>

#include <type_traits>

namespace sofa::component::mass
{

template <class DataTypes>
class UniformMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformMass,DataTypes),
               SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    using TMassType = typename sofa::component::mass::MassType<DataTypes>::type;
    
    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef type::vector<Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices DataSetIndex;
    typedef TMassType MassType;

    Data<MassType> d_vertexMass;   ///< single value defining the mass of each particle
    Data<SReal> d_totalMass;    ///< if >0 : total mass of this body
    sofa::core::objectmodel::DataFileName d_filenameMass; ///< a .rigid file to automatically load the inertia matrix and other parameters

    Data<bool>  d_showCenterOfGravity; ///< to display the center of gravity of the system
    Data<float> d_showAxisSize;        ///< to display the center of gravity of the system

    Data<bool>  d_computeMappingInertia; ///< to be used if the mass is placed under a mapping
    Data<bool>  d_showInitialCenterOfGravity; ///< display the initial center of gravity of the system

    Data<bool>  d_showX0; ///< to display the rest positions

    /// optional range of local DOF indices. Any computation involving only
    /// indices outside of this range are discarded (useful for parallelization
    /// using mesh partitionning)
    Data< type::Vec<2,int> > d_localRange;
    DataSetIndex     d_indices; ///< optional local DOF indices. Any computation involving only indices outside of this list are discarded
    Data<bool> d_preserveTotalMass; ///< Prevent totalMass from decreasing when removing particles.

    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using core::behavior::ForceField<DataTypes>::mstate ;
    using core::objectmodel::BaseObject::getContext;
    ////////////////////////////////////////////////////////////////////////////

    /// Link to be set to the topology container in the component graph.
    SingleLink <UniformMass<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    UniformMass();

    ~UniformMass();

    /// @internal fonction called in the constructor that can be specialized
    void constructor_message() ;

public:

    /// @name Read and write access functions in mass information
    /// @{
    void setMass(const MassType& d_vertexMass);
    const MassType& getVertexMass() const { return d_vertexMass.getValue(); }
    const MassType& getMass() const { return this->getVertexMass(); }

    SReal getTotalMass() const { return d_totalMass.getValue(); }
    void setTotalMass(SReal m);
    /// }@

    void setFileMass(const std::string& file) {d_filenameMass.setValue(file);}
    std::string getFileMass() const {return d_filenameMass.getFullPath();}

    void loadRigidMass(const std::string& filename);

    void reinit() override;
    void init() override;
    void initDefaultImpl() ;
    void doUpdateInternal() override;

    /// @name Check and standard initialization functions from mass information
    /// @{
    virtual bool checkVertexMass();
    virtual void initFromVertexMass();

    virtual bool checkTotalMass();
    virtual void checkTotalMassInit();
    virtual void initFromTotalMass();
    /// @}

    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor) override;
    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f) override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& d_v) const override;  ///< vMv/2 using dof->getV() override
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;   ///< Mgx potential in a uniform gravity field, null at origin
    type::Vec6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const override;  ///< (Mv,cross(x,Mv)+Iw) override

    void addMDxToVector(linearalgebra::BaseVector *resVect, const VecDeriv *dx, SReal mFact, unsigned int& offset);

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v) override;

    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override; /// Add Mass contribution to global Matrix assembling
    void buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* /* matrix */) override {}
    void buildDampingMatrix(core::behavior::DampingMatrix* /* matrices */) override {}

    SReal getElementMass(sofa::Index index) const override;
    void getElementMass(sofa::Index index, linearalgebra::BaseMatrix *m) const override;

    bool isDiagonal() const override {return true;}

    void draw(const core::visual::VisualParams* vparams) override;

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override
    {
        Inherited::parse(arg);
        parseMassTemplate<MassType>(arg, this);
    }

private:
    void updateMassOnResize(sofa::Size newSize);

    template<class T>
    void drawRigid3DImpl(const core::visual::VisualParams* vparams) ;

    template<class T>
    void drawRigid2DImpl(const core::visual::VisualParams* vparams) ;

    template<class T>
    void drawVec6Impl(const core::visual::VisualParams* vparams) ;

    template<class T>
    SReal getPotentialEnergyRigidImpl(const core::MechanicalParams* mparams,
                                      const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin



    template<class T>
    type::Vec6 getMomentumRigid3DImpl(const core::MechanicalParams* mparams,
                                                const DataVecCoord& x,
                                                const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)
    template<class T>
    type::Vec6 getMomentumVec3DImpl(const core::MechanicalParams* mparams,
                                              const DataVecCoord& x,
                                              const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)

    template <class T>
    void loadFromFileRigidImpl(const std::string& filename) ;

    template <class T>
    void addMDxToVectorVecImpl(linearalgebra::BaseVector *resVect,
                               const VecDeriv* dx,
                               SReal mFact,
                               unsigned int& offset);

};

// Specialization for rigids
template <>
void UniformMass<defaulttype::Rigid3Types>::init();
template <>
void UniformMass<defaulttype::Rigid3Types>::loadRigidMass ( const std::string&  );
template <>
void UniformMass<defaulttype::Rigid3Types>::draw(const core::visual::VisualParams* vparams);
template <>
void UniformMass<defaulttype::Rigid2Types>::draw(const core::visual::VisualParams* vparams);
template <>
SReal UniformMass<defaulttype::Rigid3Types>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& x ) const;
template <>
SReal UniformMass<defaulttype::Rigid2Types>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& x ) const;
template <>
void UniformMass<defaulttype::Vec6Types>::draw(const core::visual::VisualParams* vparams);

#if !defined(SOFA_COMPONENT_MASS_UNIFORMMASS_CPP)
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MASS_API UniformMass<defaulttype::Rigid2Types>;
#endif

} // namespace sofa::component::mass

