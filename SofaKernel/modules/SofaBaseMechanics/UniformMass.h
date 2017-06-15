/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_H
#define SOFA_COMPONENT_MASS_UNIFORMMASS_H
#include "config.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class TMassType>
class UniformMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(UniformMass,DataTypes,TMassType),
               SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef TMassType MassType;

    Data<MassType>                        d_mass;         ///< the mass of each particle
    Data<SReal>                           d_totalMass;    ///< if >0 : total mass of this body
    sofa::core::objectmodel::DataFileName d_filenameMass; ///< a .rigid file to automatically load the inertia matrix and other parameters

    Data<bool>                            d_showCenterOfGravity; /// to display the center of gravity of the system
    Data<float>                           d_showAxisSize;        /// to display the center of gravity of the system

    Data<bool>  d_computeMappingInertia;
    Data<bool>  d_showInitialCenterOfGravity;

    Data<bool>  d_showX0; /// to display the rest positions

    /// optional range of local DOF indices. Any computation involving only
    /// indices outside of this range are discarded (useful for parallelization
    /// using mesh partitionning)
    Data< defaulttype::Vec<2,int> > d_localRange;
    Data< helper::vector<int> >     d_indices;

    Data<bool> d_handleTopoChange;
    Data<bool> d_preserveTotalMass;

    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using core::behavior::ForceField<DataTypes>::mstate ;
    using core::objectmodel::BaseObject::getContext;
    ////////////////////////////////////////////////////////////////////////////

    bool m_doesTopoChangeAffect;


protected:
    UniformMass();

    ~UniformMass();

    /// @internal fonction called in the constructor that can be specialized
    void constructor_message() ;

public:
    void setMass(const MassType& d_mass);
    const MassType& getMass() const { return d_mass.getValue(); }

    SReal getTotalMass() const { return d_totalMass.getValue(); }
    void setTotalMass(SReal m);

    void setFileMass(const std::string& file) {d_filenameMass.setValue(file);}
    std::string getFileMass() const {return d_filenameMass.getFullPath();}

    void loadRigidMass(const std::string& filename);
    // -- Mass interface

    void reinit();
    void init();

    void handleTopologyChange();
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& d_v) const;  ///< vMv/2 using dof->getV()

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    defaulttype::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)

    void addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv *dx, SReal mFact, unsigned int& offset);

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v);

    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix); /// Add Mass contribution to global Matrix assembling

    SReal getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

    bool isDiagonal() {return true;}

    void draw(const core::visual::VisualParams* vparams);

private:
    template<class T>
    void reinitDefaultImpl() ;

    template<class T>
    void reinitRigidImpl() ;

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
    defaulttype::Vector6 getMomentumRigid3DImpl(const core::MechanicalParams* mparams,
                                                const DataVecCoord& x,
                                                const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)
    template<class T>
    defaulttype::Vector6 getMomentumVec3DImpl(const core::MechanicalParams* mparams,
                                              const DataVecCoord& x,
                                              const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)

    template <class T>
    void loadFromFileRigidImpl(const std::string& filename) ;

    template <class T>
    void addMDxToVectorVecImpl(defaulttype::BaseVector *resVect,
                               const VecDeriv* dx,
                               SReal mFact,
                               unsigned int& offset);

};

//Specialization for rigids
#ifdef SOFA_WITH_DOUBLE
template <>
void UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::reinit();
template <>
void UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::loadRigidMass ( const std::string&  );
template <>
void UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>::draw(const core::visual::VisualParams* vparams);
template <>
void UniformMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>::draw(const core::visual::VisualParams* vparams);
template <>
double UniformMass<defaulttype::Rigid3dTypes,defaulttype::Rigid3dMass>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& x ) const;
template <>
double UniformMass<defaulttype::Rigid2dTypes,defaulttype::Rigid2dMass>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& x ) const;
template <>
void UniformMass<defaulttype::Vec6dTypes,double>::draw(const core::visual::VisualParams* vparams);
#endif
#ifdef SOFA_WITH_FLOAT
template<>
void UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::reinit();
template<>
void UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::loadRigidMass ( const std::string& );
template <>
void UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>::draw(const core::visual::VisualParams* vparams);
template <>
void UniformMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>::draw(const core::visual::VisualParams* vparams);
template <>
SReal UniformMass<defaulttype::Rigid3fTypes,defaulttype::Rigid3fMass>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& x ) const;
template <>
SReal UniformMass<defaulttype::Rigid2fTypes,defaulttype::Rigid2fMass>::getPotentialEnergy ( const core::MechanicalParams*, const DataVecCoord& x ) const;
template <>
void UniformMass<defaulttype::Vec6fTypes,float>::draw(const core::visual::VisualParams* vparams);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MASS_UNIFORMMASS_CPP)
#ifdef SOFA_WITH_DOUBLE
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec3dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec2dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec1dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec6dTypes, double>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid3dTypes, defaulttype::Rigid3dMass>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid2dTypes, defaulttype::Rigid2dMass>;
#endif
#ifdef SOFA_WITH_FLOAT
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec3fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec2fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec1fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Vec6fTypes, float>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid3fTypes, defaulttype::Rigid3fMass>;
extern template class SOFA_BASE_MECHANICS_API UniformMass<defaulttype::Rigid2fTypes, defaulttype::Rigid2fMass>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif

