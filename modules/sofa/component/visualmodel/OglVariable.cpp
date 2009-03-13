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
#include <sofa/component/visualmodel/OglVariable.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/** SINGLE INT VARIABLE **/
SOFA_DECL_CLASS(OglIntVariable)
SOFA_DECL_CLASS(OglInt2Variable)
SOFA_DECL_CLASS(OglInt3Variable)
SOFA_DECL_CLASS(OglInt4Variable)

//Register OglIntVariable in the Object Factory
int OglIntVariableClass = core::RegisterObject("OglIntVariable")
        .add< OglIntVariable >()
        ;
//Register OglInt2Variable in the Object Factory
int OglInt2VariableClass = core::RegisterObject("OglInt2Variable")
        .add< OglInt2Variable >()
        ;
//Register OglInt3Variable in the Object Factory
int OglInt3VariableClass = core::RegisterObject("OglInt3Variable")
        .add< OglInt3Variable >()
        ;
//Register OglInt4Variable in the Object Factory
int OglInt4VariableClass = core::RegisterObject("OglInt4Variable")
        .add< OglInt4Variable >()
        ;

/** SINGLE FLOAT VARIABLE **/

SOFA_DECL_CLASS(OglFloatVariable)
SOFA_DECL_CLASS(OglFloat2Variable)
SOFA_DECL_CLASS(OglFloat3Variable)
SOFA_DECL_CLASS(OglFloat4Variable)

//Register OglFloatVariable in the Object Factory
int OglFloatVariableClass = core::RegisterObject("OglFloatVariable")
        .add< OglFloatVariable >()
        ;
//Register OglFloat2Variable in the Object Factory
int OglFloat2VariableClass = core::RegisterObject("OglFloat2Variable")
        .add< OglFloat2Variable >()
        ;
//Register OglFloat3Variable in the Object Factory
int OglFloat3VariableClass = core::RegisterObject("OglFloat3Variable")
        .add< OglFloat3Variable >()
        ;
//Register OglFloat4Variable in the Object Factory
int OglFloat4VariableClass = core::RegisterObject("OglFloat4Variable")
        .add< OglFloat4Variable >()
        ;

/** INT VECTOR VARIABLE **/
SOFA_DECL_CLASS(OglIntVectorVariable)
SOFA_DECL_CLASS(OglIntVector2Variable)
SOFA_DECL_CLASS(OglIntVector3Variable)
SOFA_DECL_CLASS(OglIntVector4Variable)

//Register OglIntVectorVariable in the Object Factory
int OglIntVectorVariableClass = core::RegisterObject("OglIntVectorVariable")
        .add< OglIntVectorVariable >()
        ;

//Register OglIntVector2Variable in the Object Factory
int OglIntVector2VariableClass = core::RegisterObject("OglIntVector2Variable")
        .add< OglIntVector2Variable >()
        ;

//Register OglIntVector3Variable in the Object Factory
int OglIntVector3VariableClass = core::RegisterObject("OglIntVector3Variable")
        .add< OglIntVector3Variable >()
        ;

//Register OglIntVector4Variable in the Object Factory
int OglIntVector4VariableClass = core::RegisterObject("OglIntVector4Variable")
        .add< OglIntVector4Variable >()
        ;


/** FLOAT VECTOR VARIABLE **/
SOFA_DECL_CLASS(OglFloatVectorVariable)
SOFA_DECL_CLASS(OglFloatVector2Variable)
SOFA_DECL_CLASS(OglFloatVector3Variable)
SOFA_DECL_CLASS(OglFloatVector4Variable)

//Register OglFloatVectorVariable in the Object Factory
int OglFloatVectorVariableClass = core::RegisterObject("OglFloatVectorVariable")
        .add< OglFloatVectorVariable >()
        ;

//Register OglFloatVector2Variable in the Object Factory
int OglFloatVector2VariableClass = core::RegisterObject("OglFloatVector2Variable")
        .add< OglFloatVector2Variable >()
        ;

//Register OglFloatVector3Variable in the Object Factory
int OglFloatVector3VariableClass = core::RegisterObject("OglFloatVector3Variable")
        .add< OglFloatVector3Variable >()
        ;
//Register OglFloatVector4Variable in the Object Factory
int OglFloatVector4VariableClass = core::RegisterObject("OglFloatVector4Variable")
        .add< OglFloatVector4Variable >()
        ;


}

}

}
