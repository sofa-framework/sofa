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
#include <sofa/component/constraint/lagrangian/correction/config.h>

#include <sofa/core/behavior/ConstraintCorrection.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/linearalgebra/FullMatrix.h>

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

namespace sofa::component::constraint::lagrangian::correction
{

/**
 *  \brief Component computing constraint forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class PrecomputedConstraintCorrection : public sofa::core::behavior::ConstraintCorrection< TDataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PrecomputedConstraintCorrection,TDataTypes), SOFA_TEMPLATE(core::behavior::ConstraintCorrection, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;

    typedef sofa::core::behavior::ConstraintCorrection< TDataTypes > Inherit;

    typedef typename Coord::value_type Real;
    typedef sofa::type::MatNoInit<3, 3, Real> Transformation;

    Data<bool> m_rotations;
    Data<bool> m_restRotations;

    Data<bool> recompute; ///< if true, always recompute the compliance
	Data<SReal> debugViewFrameScale; ///< Scale on computed node's frame
	sofa::core::objectmodel::DataFileName f_fileCompliance; ///< Precomputed compliance matrix data file
	Data<std::string> fileDir; ///< If not empty, the compliance will be saved in this repertory
    
protected:
    PrecomputedConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm = nullptr);

    virtual ~PrecomputedConstraintCorrection();
public:
    void bwdInit() override;

    void addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, sofa::linearalgebra::BaseMatrix* W) override;

    void getComplianceMatrix(linearalgebra::BaseMatrix* m) const override;

    void computeMotionCorrection(const core::ConstraintParams*, core::MultiVecDerivId dx, core::MultiVecDerivId f) override;

    void applyMotionCorrection(const sofa::core::ConstraintParams *cparams, sofa::Data< VecCoord > &x, sofa::Data< VecDeriv > &v, sofa::Data< VecDeriv > &dx , const sofa::Data< VecDeriv > & correction) override;

    void applyPositionCorrection(const sofa::core::ConstraintParams *cparams, sofa::Data< VecCoord > &x, sofa::Data< VecDeriv > &dx, const sofa::Data< VecDeriv > & correction) override;

    void applyVelocityCorrection(const sofa::core::ConstraintParams *cparams, sofa::Data< VecDeriv > &v, sofa::Data< VecDeriv > &dv, const sofa::Data< VecDeriv > & correction) override;

    /// @name Deprecated API
    /// @{

    void applyContactForce(const linearalgebra::BaseVector *f) override;

    void resetContactForce() override;

    /// @}

    virtual void rotateConstraints(bool back);

    virtual void rotateResponse();

    void draw(const core::visual::VisualParams* vparams) override;

    /// @name Unbuilt constraint system during resolution
    /// @{

    void resetForUnbuiltResolution(SReal* f, std::list<unsigned int>& /*renumbering*/) override;

    bool hasConstraintNumber(int index) override;  // virtual ???

    void addConstraintDisplacement(SReal* d, int begin,int end) override;

    void setConstraintDForce(SReal* df, int begin, int end, bool update) override;

    void getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W, int begin, int end) override;

    /// @}

public:

    struct InverseStorage
    {
        Real* data;
        int nbref;
        InverseStorage() : data(nullptr), nbref(0) {}
    };

    std::string invName;
    InverseStorage* invM;
    Real* appCompliance;
    unsigned int dimensionAppCompliance;

    static std::map<std::string, InverseStorage>& getInverseMap()
    {
        static std::map<std::string, InverseStorage> registry;
        return registry;
    }

    static InverseStorage* getInverse(std::string name);

    static void releaseInverse(std::string name, InverseStorage* inv);

    unsigned int nbRows, nbCols, dof_on_node, nbNodes;
    type::vector<int> _indexNodeSparseCompliance;
    type::vector<Deriv> _sparseCompliance;
    Real Fbuf[6], DXbuf;

    // new :  for non building the constraint system during solving process //
    //VecDeriv constraint_disp, constraint_force;
    type::vector<int> id_to_localIndex;	// table that gives the local index of a constraint given its id
    type::vector<unsigned int> localIndex_to_id; //inverse table that gives the id of a constraint given its local index
    std::list<unsigned int> active_local_force; // table of local index of the non-null forces;
    linearalgebra::FullMatrix< Real > localW;
    SReal* constraint_force;

    // NEW METHOD FOR UNBUILT
    // new :  for non building the constraint system during solving process //
    VecDeriv constraint_D, constraint_F;
    std::list<int> constraint_dofs;		// list of indices of each point which is involve with constraint

public:
    Real* getInverse()
    {
        if (invM->data)
            return invM->data;
        else
            msg_error() << "Inverse is not computed yet";
        return nullptr;
    }

protected:
    /**
     * @brief Load compliance matrix from memory or external file according to fileName.
     *
     * @return Loading success.
     */
    bool loadCompliance(std::string fileName);

    /**
     * @brief Save compliance matrix into a file.
     */
    void saveCompliance(const std::string& fileName);

    /**
     * @brief Builds the compliance file name using the SOFA component internal data.
     */
    std::string buildFileName();

    /**
     * @brief Compute dx correction from motion space force vector.
     */
    void computeDx(Data<VecDeriv>& dx, const Data< VecDeriv > &f, const std::list< int > &activeDofs);

    std::list< int > m_activeDofs;
};



template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3Types>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1Types>::rotateConstraints(bool back);

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3Types>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1Types>::rotateResponse();

template<>
void PrecomputedConstraintCorrection<defaulttype::Rigid3Types>::draw(const core::visual::VisualParams* vparams);

template<>
void PrecomputedConstraintCorrection<defaulttype::Vec1Types>::draw(const core::visual::VisualParams* vparams);



#if !defined(SOFA_COMPONENT_CONSTRAINTSET_PRECOMPUTEDCONSTRAINTCORRECTION_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API PrecomputedConstraintCorrection<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API PrecomputedConstraintCorrection<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API PrecomputedConstraintCorrection<defaulttype::Rigid3Types>;

#endif


} //namespace sofa::component::constraint::lagrangian::correction
