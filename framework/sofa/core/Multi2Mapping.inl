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
#ifndef SOFA_CORE_MULTI2MAPPING_INL
#define SOFA_CORE_MULTI2MAPPING_INL

#include <sofa/core/Multi2Mapping.h>

namespace sofa
{

namespace core
{



template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addInputModel(In1* from)
{
    if (!from) return;
    this->fromModels1.push_back(from);
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addInputModel(In2* from)
{
    if (!from) return;
    this->fromModels2.push_back(from);
}

template< class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::addOutputModel(Out* to)
{
    this->toModels.insert( toModels.end(), to);
}

template< class In1, class In2, class Out > template <class In>
helper::vector<In*>&  Multi2Mapping<In1,In2,Out>::getFromModels()
{
    if (!fromModels1.empty() && dynamic_cast<In*>(fromModels1[0]))
    {
        return fromModels1;
    }
    else if (!fromModels2.empty() && dynamic_cast<In*>(fromModels2[0]))
    {
        return fromModels2;
    }
}

template< class In1, class In2, class Out >
helper::vector<Out*>& Multi2Mapping<In1,In2,Out>::getToModels()
{
    return toModels;
}
template< class In1, class In2, class Out >
helper::vector<objectmodel::BaseObject*> Multi2Mapping<In1,In2,Out>::getFrom()
{
    helper::vector<objectmodel::BaseObject*> base_fromModels;
    std::copy(fromModels1.begin(), fromModels1.end(), std::back_inserter(base_fromModels));
    std::copy(fromModels2.begin(), fromModels2.end(), std::back_inserter(base_fromModels));
    return base_fromModels;
}

template< class In1, class In2, class Out >
helper::vector<objectmodel::BaseObject* > Multi2Mapping<In1,In2,Out>::getTo()
{
    helper::vector<objectmodel::BaseObject*> base_toModels;
    std::copy(toModels.begin(), toModels.end(), std::back_inserter(base_toModels));
    return base_toModels;
}
template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::init()
{
    this->updateMapping();
}


template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::updateMapping()
{
    if( (this->fromModels1.empty() && this->fromModels2.empty() ) || this->toModels.empty() )
        return;

    const VecId &idCoord = VecId::position();
    helper::vector<typename Out::VecCoord*> vecOutPos;
    getVecOutCoord(idCoord, vecOutPos);

    helper::vector<const typename In1::VecCoord*> vecIn1Pos;
    getConstVecIn1Coord(idCoord, vecIn1Pos);

    helper::vector<const typename In2::VecCoord*> vecIn2Pos;
    getConstVecIn2Coord(idCoord, vecIn2Pos);

    apply ( vecOutPos, vecIn1Pos, vecIn2Pos);


    const VecId &idDeriv = VecId::velocity();
    helper::vector<typename Out::VecDeriv*> vecOutVel;
    getVecOutDeriv(idDeriv, vecOutVel);
    helper::vector<const typename In1::VecDeriv*> vecIn1Vel;
    getConstVecIn1Deriv(idDeriv, vecIn1Vel);
    helper::vector<const typename In2::VecDeriv*> vecIn2Vel;
    getConstVecIn2Deriv(idDeriv, vecIn2Vel);
    applyJ( vecOutVel, vecIn1Vel, vecIn2Vel);
}


template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getVecIn1Coord(const VecId &id, helper::vector<typename In1::VecCoord*> &v) const
{
    for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(fromModels1[i]->getVecCoord(id.index));
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getVecIn1Deriv(const VecId &id, helper::vector<typename In1::VecDeriv*> &v) const
{
    for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(fromModels1[i]->getVecDeriv(id.index));
}


template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getVecIn2Coord(const VecId &id, helper::vector<typename In2::VecCoord*> &v) const
{
    for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecCoord(id.index));
}


template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getVecIn2Deriv(const VecId &id, helper::vector<typename In2::VecDeriv*> &v) const
{
    for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecDeriv(id.index));
}


template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getConstVecIn2Deriv(const VecId &id, helper::vector<const typename In2::VecDeriv*> &v) const
{
    for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(fromModels2[i]->getVecDeriv(id.index));
}





template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getVecOutDeriv(const VecId &id, helper::vector<typename Out::VecDeriv*> &v) const
{
    for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecDeriv(id.index));
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getConstVecOutCoord(const VecId &id, helper::vector<const typename Out::VecCoord*> &v) const
{
    for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecCoord(id.index));
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::getConstVecOutDeriv(const VecId &id, helper::vector<const typename Out::VecDeriv*> &v) const
{
    for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(toModels[i]->getVecDeriv(id.index));
}



template < class In1, class In2, class Out >
std::string Multi2Mapping<In1,In2,Out>::templateName(const Multi2Mapping<In1, In2, Out>* /*mapping*/)
{
    return std::string("Multi2Mapping<[")+In1::DataTypes::Name() + std::string(",") +In2::DataTypes::Name() + std::string("],")+ Out::DataTypes::Name() + std::string(">");
}

template < class In1, class In2, class Out >
void Multi2Mapping<In1,In2,Out>::disable()
{
}


} //core
} //sofa

#endif
