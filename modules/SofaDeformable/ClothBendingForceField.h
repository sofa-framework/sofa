/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHBENDINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_CLOTHBENDINGFORCEFIELD_H

#define DEBUG

#include <SofaDeformable/StiffSpringForceField.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/***
This force field implement a bending force presented in the paper of Choi and al 
"Stable but Responsive Cloth" (http://graphics.snu.ac.kr/~kjchoi/publication/cloth.pdf)
This force field improve the stability of the system by ensuring that the stiffness matrix 
remains positive definite and is well suited for cloth simulation.
*/
template<class DataTypes>
class ClothBendingForceField : public StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ClothBendingForceField, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    typedef StiffSpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef typename Inherit::Spring Spring;


    Data<SReal> kb; // flexural rigidity 
    Data<bool> debug;

protected:
    typedef std::pair<unsigned,unsigned> IndexPair;

    void addSpringForce(Real& potentialEnergy, VecDeriv& f1, const  VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const  VecCoord& p2, const  VecDeriv& v2, int i, const Spring& spring);

    void addSpring( unsigned a, unsigned b, std::set<IndexPair>& springSet );
    
    ClothBendingForceField();
    virtual ~ClothBendingForceField();

    // the following functions represents a polynomial approximation (and its derivative) of the the bending force intensity described in the paper above
    // This polynom was computed using maple.
    inline Real fb(const Real length, const Real restLength)
    {
        Real x      = length/restLength;
        Real xx     = x*x;
        Real xxxx   = xx*xx;
        return (Real)(-0.3416484836e2 + 0.6759797376e2 * x - 0.8277725454e2 * xx + 0.5165991237e2 * xx * x - 0.1433616469e2 * xxxx);

    }

    inline Real dfb(const Real length, const Real restLength)
    {
        Real x  = length/restLength;
        Real xx = x*x;
        return (Real) (0.6759797376e2 - 0.1655545091e3 * x + 0.1549797371e3 * xx - 0.5734465876e2 * xx * x);
    }

    // this function searches an edge in the continuity of one given in parameter and return true in case of success 
    bool findAlignedEdges(const unsigned index1, const unsigned index2, unsigned& index3);

    // debugging informations
    helper::vector<Coord> debug_forces; 
    helper::vector<defaulttype::Vec4f> debug_colors;

public:

    virtual void init();
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f1, DataVecDeriv& f2, const DataVecCoord& x1, const DataVecCoord& x2, const DataVecDeriv& v1, const DataVecDeriv& v2);
    virtual void draw(const core::visual::VisualParams* vparams);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_CLOTHBENDINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_DEFORMABLE_API ClothBendingForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_DEFORMABLE_API ClothBendingForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_MESHSPRINGFORCEFIELD_H */
