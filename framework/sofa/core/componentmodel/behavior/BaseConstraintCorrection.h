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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINTCORRECTION_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINTCORRECTION_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/// @TODO All methods in this class need to be commented

/**
 *  \brief Component computing contact forces within a simulated body using the compliance method.
 */
class BaseConstraintCorrection : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseConstraintCorrection, BaseObject);

    virtual ~BaseConstraintCorrection() {}

    virtual void getCompliance(defaulttype::BaseMatrix* W) = 0;



    virtual void CudaGetCompliance(defaulttype::BaseMatrix* W)
    {
        sout << "warning : CudaGetCompliance(defaulttype::BaseMatrix* W) is not implemented in " << this->getTypeName() << sendl;
        getCompliance(W); // par defaut si la methode cuda n'est pas implementé on resoud sur CPU
    }

    virtual void applyContactForce(const defaulttype::BaseVector *f) = 0;

    virtual void resetContactForce() = 0;

    // NEW : for non building the constraint system during solving/////////////////
    virtual bool hasConstraintNumber(int /*index*/) {return true;}

    virtual void resetForUnbuiltResolution(double * /*f*/, std::list<int>& /*renumbering*/) {}

    virtual void addConstraintDisplacement(double * /*d*/, int /*begin*/,int /*end*/) { }

    virtual void setConstraintDForce(double * /*df*/, int /*begin*/,int /*end*/, bool /*update*/) { }	  // f += df

    virtual void getBlockDiagonalCompliance(defaulttype::BaseMatrix* /*W*/, int /*begin*/,int /*end*/)
    {
        sout << "warning : getBlockDiagonalCompliance(defaulttype::BaseMatrix* W) is not implemented in " << this->getTypeName() << sendl;
    }
    /////////////////////////////////////////////////////////////////////////////////

};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
