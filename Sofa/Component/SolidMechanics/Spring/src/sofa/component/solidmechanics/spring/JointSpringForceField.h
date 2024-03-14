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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/PairInteractionForceField.h>

namespace sofa::component::solidmechanics::spring
{

//This class is specialized with cudatypes
template<class DataTypes>
class JointSpringForceFieldInternalData
{
public:
};
  
template<typename DataTypes>
class JointSpring;


/** JointSpringForceField simulates 6D springs between Rigid DOFS
  Use kst vector to specify the directionnal stiffnesses (on each local axe)
  Use ksr vector to specify the rotational stiffnesses (on each local axe)
*/
template<class DataTypes>
class JointSpringForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(JointSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=DataTypes::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;
    typedef type::Vec<N,Real> Vector;

    typedef JointSpring<DataTypes> Spring;

protected:

    SReal m_potentialEnergy;
    Real m_lastTime;

    std::ifstream* m_infile;
    std::ofstream* m_outfile;

    JointSpringForceFieldInternalData<DataTypes> data;
    friend class JointSpringForceFieldInternalData<DataTypes>;

    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce(SReal& potentialEnergy,
                        VecDeriv& f1,
                        const VecCoord& p1,
                        const VecDeriv& v1,
                        VecDeriv& f2,
                        const VecCoord& p2,
                        const VecDeriv& v2,
                        sofa::Index i, /*const*/ Spring& spring);

    /// Apply the stiffness, i.e. accumulate df given dx
    void addSpringDForce(VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, sofa::Index i, /*const*/ Spring& spring, Real kFactor);

    // project torsion to Lawfulltorsion according to limitangles
    void projectTorsion(Spring& spring);



    JointSpringForceField(MechanicalState* object1, MechanicalState* object2);
    JointSpringForceField();

    virtual ~JointSpringForceField();

public:

    ////////////////////////// Inherited from BaseObject /////////////////////////
    void init() override;
    void bwdInit() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams*  params, bool /*onlyVisible*/) override;


    ///////////////////////// Inherited from PairInteractionForceField ///////////////////
    void addForce(  const core::MechanicalParams* mparams,
                            DataVecDeriv& data_f1,
                            DataVecDeriv& data_f2,
                            const DataVecCoord& data_x1,
                            const DataVecCoord& data_x2,
                            const DataVecDeriv& data_v1,
                            const DataVecDeriv& data_v2 ) override;

    void addDForce(const core::MechanicalParams* mparams,
                           DataVecDeriv& data_df1,
                           DataVecDeriv& data_df2,
                           const DataVecDeriv& data_dx1,
                           const DataVecDeriv& data_dx2) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams*,
                                     const DataVecCoord&,
                                     const DataVecCoord& ) const override { return m_potentialEnergy; }


    //////////////////////////   Data fields    //////////////////////////////////
    /// the list of the springs
    sofa::core::objectmodel::DataFileName f_outfilename; ///< output file name
    sofa::core::objectmodel::DataFileName f_infilename; ///< input file containing constant joint force
    Data <Real > f_period; ///< period between outputs
    Data<bool> f_reinit; ///< flag enabling reinitialization of the output file at each timestep
    Data<sofa::type::vector<Spring> > d_springs; ///< pairs of indices, stiffness, damping, rest length

    /// bool to allow the display of the 2 parts of springs torsions
    Data<bool> d_showLawfulTorsion;
    Data<bool> d_showExtraTorsion; ///< display the illicit part of the joint rotation
    Data<Real> d_showFactorSize; ///< modify the size of the debug information of a given factor


    ///////////////////////////////////////////////////////////////////////////////
    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }
    sofa::type::vector<Spring> * getSprings() { return d_springs.beginEdit(); }

    // -- Modifiers
    void clear(sofa::Size reserve=0) ;
    void addSpring(const Spring& s) ;
    void addSpring(sofa::Index m1, sofa::Index m2, Real softKst, Real hardKst, Real softKsr,
                   Real hardKsr, Real blocKsr, Real axmin, Real axmax, Real aymin,
                   Real aymax, Real azmin, Real azmax, Real kd);


};

#if !defined(SOFA_COMPONENT_FORCEFIELD_JOINTSPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API JointSpringForceField<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
