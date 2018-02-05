/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_VECTORSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_VECTORSPRINGFORCEFIELD_H
#include "config.h"

#include <SofaDeformable/SpringForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>
#include <sofa/core/MechanicalParams.h>


namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
class VectorSpringForceField: public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VectorSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    struct Spring
    {
        Real  ks;          ///< spring stiffness
        Real  kd;          ///< damping factor
        Deriv restVector;  ///< rest vector of the spring

        Spring(Real _ks, Real _kd, Deriv _rl) : ks(_ks), kd(_kd), restVector(_rl)
        {
        }
        Spring() : ks(1.0), kd(1.0)
        {
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const Spring& /*s*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, Spring& /*s*/ )
        {
            return in;
        }
    };
protected:

    SReal m_potentialEnergy;
    /// true if the springs are initialized from the topology
    bool useTopology;
    bool usingMask;
    /// indices in case we don't use the topology
    sofa::helper::vector<core::topology::BaseMeshTopology::Edge> edgeArray;


    void resizeArray(unsigned int n);


public:

    /// where the springs information are stored
    sofa::component::topology::EdgeData<sofa::helper::vector<Spring> > springArray;

    class EdgeDataHandler : public sofa::component::topology::TopologyDataHandler< core::topology::BaseMeshTopology::Edge, sofa::helper::vector<Spring> >
    {
    public:
        typedef typename VectorSpringForceField<DataTypes>::Spring Spring;
        EdgeDataHandler(VectorSpringForceField<DataTypes>* ff, topology::EdgeData<sofa::helper::vector<Spring> >* data)
            :topology::TopologyDataHandler< core::topology::BaseMeshTopology::Edge,sofa::helper::vector<Spring> >(data)
            ,ff(ff)
        {

        }

        void applyCreateFunction(unsigned int, Spring &t,
                const core::topology::BaseMeshTopology::Edge &,
                const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &);
    protected:
        VectorSpringForceField<DataTypes>* ff;

    };
    /// the filename where to load the spring information
    sofa::core::objectmodel::DataFileName m_filename;
    /// By default, assume that all edges have the same stiffness
    Data<double> m_stiffness;
    /// By default, assume that all edges have the same viscosity
    Data<double> m_viscosity;

    Data<bool> m_useTopology;

    sofa::core::topology::BaseMeshTopology* _topology;
    sofa::component::topology::EdgeSetTopologyContainer* edgeCont;
    sofa::component::topology::EdgeSetGeometryAlgorithms<DataTypes>* edgeGeo;
    sofa::component::topology::EdgeSetTopologyModifier* edgeMod;
protected:
    VectorSpringForceField(MechanicalState* _object=NULL);

    VectorSpringForceField(MechanicalState* _object1, MechanicalState* _object2);
    virtual ~VectorSpringForceField();

    EdgeDataHandler* edgeHandler;

public:
    bool load(const char *filename);

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    virtual void init() override;
    virtual void bwdInit() override;

    void createDefaultSprings();

    virtual void handleEvent( core::objectmodel::Event* e ) override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;

    virtual SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const override { return m_potentialEnergy; }

    Real getStiffness() const
    {
        return (Real)(m_stiffness.getValue());
    }
    const Real getViscosity() const
    {
        return (Real)(m_viscosity.getValue());
    }
    const topology::EdgeData<sofa::helper::vector<Spring> >& getSpringArray() const
    {
        return springArray;
    }

    void draw(const core::visual::VisualParams* vparams) override;

    // -- Modifiers

    void clear(int reserve=0)
    {
        helper::vector<Spring>& springArrayData = *(springArray.beginEdit());
        springArrayData.clear();
        if (reserve) springArrayData.reserve(reserve);
        springArray.endEdit();
        if(!useTopology) edgeArray.clear();
    }

    void addSpring(int m1, int m2, SReal ks, SReal kd, Coord restVector);

    /// forward declaration of the loader class used to read spring information from file
    class Loader;
    friend class Loader;

    virtual void updateForceMask() override;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_VECTORSPRINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_DEFORMABLE_API VectorSpringForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_DEFORMABLE_API VectorSpringForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_VECTORSPRINGFORCEFIELD_H */
