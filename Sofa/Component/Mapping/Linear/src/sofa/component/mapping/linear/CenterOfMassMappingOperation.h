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

namespace sofa::component::mapping::linear {

template<typename Model>
struct CenterOfMassMappingOperation
{
    typedef typename Model::VecCoord VecCoord;
    typedef typename Model::Coord Coord;
    typedef typename Model::Deriv Deriv;
    typedef typename Model::VecDeriv VecDeriv;

public :
    static inline const VecCoord *getVecCoord(const Model *m, const sofa::core::VecId id) {
        return m->getVecCoord(id.index);
    }

    static inline VecDeriv *getVecDeriv(Model *m, const sofa::core::VecId id) { return m->getVecDeriv(id.index); }

    static inline const sofa::core::behavior::BaseMass *fetchMass(const Model *m) {
        const sofa::core::behavior::BaseMass *mass = m->getContext()->getMass();
        return mass;
    }

    static inline double computeTotalMass(const Model *model, const sofa::core::behavior::BaseMass *mass) {
        double result = 0.0;
        const unsigned int modelSize = static_cast<unsigned int>(model->getSize());
        for (unsigned int i = 0; i < modelSize; i++) {
            result += mass->getElementMass(i);
        }
        return result;
    }

    static inline Coord WeightedCoord(const VecCoord *v, const sofa::core::behavior::BaseMass *m) {
        Coord c;
        for (unsigned int i = 0; i < v->size(); i++) {
            c += (*v)[i] * m->getElementMass(i);
        }
        return c;
    }

    static inline Deriv WeightedDeriv(const VecDeriv *v, const sofa::core::behavior::BaseMass *m) {
        Deriv d;
        for (unsigned int i = 0; i < v->size(); i++) {
            d += (*v)[i] * m->getElementMass(i);
        }
        return d;
    }
};

}