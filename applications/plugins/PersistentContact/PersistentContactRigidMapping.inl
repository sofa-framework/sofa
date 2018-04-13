/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL

#include "PersistentContactRigidMapping.h"

#include <SofaRigid/RigidMapping.inl>

#include <sofa/simulation/AnimateEndEvent.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


template <class TIn, class TOut>
PersistentContactRigidMapping<TIn, TOut>::PersistentContactRigidMapping()
    : Inherit()
{
    m_inputMapping = 0;
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::beginAddContactPoint()
{
    if (!m_init)
    {
        if (this->f_printLog.getValue())
        {
            std::cout << "BeginAddContactPoint : pos = " << this->toModel->read(core::ConstVecCoordId::position())->getValue() << " before suppr the contact" << std::endl;
        }

        m_previousPoints = this->points.getValue();
        this->clear(0);
        this->toModel->resize(0);
        m_init = true;
    }
}


template <class TIn, class TOut>
int PersistentContactRigidMapping<TIn, TOut>::addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & /*baryCoords*/)
{
//	std::cout << "PersistentContactRigidMapping::addContactPointFromInputMapping()\n";

    if (this->f_printLog.getValue())
    {
        std::cout << "addContactPointFromInputMapping  Pos Ref = " << pos <<std::endl;
    }

    const typename In::VecCoord& xfrom = this->fromModel->read(core::ConstVecCoordId::position())->getValue();

    Coord posContact;
    for (unsigned int i = 0; i < 3; i++)
        posContact[i] = (Real) pos[i];

    unsigned int inputIdx = m_inputMapping->index.getValue();

    this->index.setValue(inputIdx);
    this->rigidIndexPerPoint.setValue(m_inputMapping->rigidIndexPerPoint.getValue());
    Coord x_local = xfrom[inputIdx].inverseRotate(posContact - xfrom[inputIdx].getCenter());

    this->addPoint(x_local, inputIdx);

    int index = this->points.getValue().size() -1;
    this->toModel->resize(index+1);

    return index;
}


template <class TIn, class TOut>
int PersistentContactRigidMapping<TIn, TOut>::keepContactPointFromInputMapping(const int _index)
{
    if (this->f_printLog.getValue())
    {
        std::cout << "keepContactPointFromInputMapping index = " << _index <<std::endl;
    }

    unsigned int inputIdx = m_inputMapping->index.getValue();

    this->index.setValue(inputIdx);
    this->rigidIndexPerPoint.setValue(m_inputMapping->rigidIndexPerPoint.getValue());

    if (_index > (int)m_previousPoints.size())
    {
        std::cout << "\nKeepContactPointFromInputMapping Critical Error!!!!!\n";
        return 0;
    }

    this->addPoint(m_previousPoints[_index], inputIdx);

    int index = this->points.getValue().size() -1;
    this->toModel->resize(index+1);

    return index;
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::init()
{
    this->f_listening.setValue(true);
    m_init = false;

    setDefaultValues();

    this->Inherit::init();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::bwdInit()
{
    const std::string path = this->m_nameOfInputMap.getValue();

    simulation::Node* parentNode = 0;
    parentNode = static_cast< simulation::Node* >(this->fromModel->getContext());

    if (parentNode)
    {
        helper::vector< Inherit* > inherits;

        parentNode->getTreeObjects< Inherit, helper::vector< Inherit* > >(&inherits);

        typename helper::vector< Inherit* >::const_iterator it = inherits.begin();
        typename helper::vector< Inherit* >::const_iterator itEnd = inherits.end();

        while (it != itEnd)
        {
            if ((*it)->getName() == path)
            {
                m_inputMapping = *it;
                break;
            }

            ++it;
        }
    }

    if (!m_inputMapping)
        serr << "WARNING : can not found the input mapping" << sendl;
    else
        sout << "Input mapping named " << m_inputMapping->getName() << " is found" << sendl;
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::reset()
{
    setDefaultValues();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::setDefaultValues()
{
    m_previousPosition = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    m_previousFreePosition = this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    m_previousDx.resize(m_previousFreePosition.size());
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::handleEvent(sofa::core::objectmodel::Event* ev)
{
    if (dynamic_cast< simulation::AnimateEndEvent* >(ev))
    {
        storeFreePositionAndDx();
        m_init = false;
    }
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::storeFreePositionAndDx()
{
    m_previousFreePosition = this->fromModel->read(core::ConstVecCoordId::freePosition())->getValue();
    m_previousDx = this->fromModel->read(core::ConstVecDerivId::dx())->getValue();

    if (this->f_printLog.getValue())
    {
        std::cout << "===== end of the time ste =========\n stored Free Pos : " << m_previousFreePosition << std::endl;
        std::cout << " stored DX : " << m_previousDx << std::endl;
        std::cout << "============================" << std::endl;
    }

//    this->applyLinearizedPosition();

}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::applyLinearizedPosition()
{
    Data< VecCoord > newXFree;
    Data< InVecCoord > prevXFree;
    prevXFree.setValue(m_previousFreePosition);

    this->apply(0, newXFree, prevXFree);

    // We need to apply the previous position to obtain the right linearization
    Data< VecCoord > tempValue;
    Data< InVecCoord > prevX;
    prevX.setValue(m_previousPosition);
    this->apply(0, tempValue, prevX);

    Data< VecDeriv > newDx;
    Data< InVecDeriv > prevDx;
    prevDx.setValue(m_previousDx);

    this->applyJ(0, newDx, prevDx);

    Data< VecCoord >* newPos_d = this->toModel->write(core::VecCoordId::position());
    VecCoord &newPos = *newPos_d->beginEdit();

    newPos = newXFree.getValue();

    for (unsigned int i=0; i < newPos.size(); i++)
    {
        newPos[i] += newDx.getValue()[i];
    }

    newPos_d->endEdit();
}


template <class TIn, class TOut>
void PersistentContactRigidMapping<TIn, TOut>::applyPositionAndFreePosition()
{
    applyLinearizedPosition();
    core::Mapping<TIn, TOut>::apply(0, sofa::core::VecCoordId::freePosition(), sofa::core::ConstVecCoordId::freePosition());
    core::Mapping<TIn, TOut>::applyJ(0, sofa::core::VecDerivId::velocity(), sofa::core::ConstVecDerivId::velocity());
    core::Mapping<TIn, TOut>::applyJ(0, sofa::core::VecDerivId::freeVelocity(), sofa::core::ConstVecDerivId::freeVelocity());
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTRIGIDMAPPING_INL
