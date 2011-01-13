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
#ifndef SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL

#include <sofa/component/forcefield/BuoyantForceField.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>
#include <iostream>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::topology;

template <class DataTypes>
BuoyantForceField<DataTypes>::BuoyantForceField():
    m_minBox(initData(&m_minBox, Coord(-100.0, -100,-100.0), "min", "Lower bound of the liquid box")),
    m_maxBox(initData(&m_maxBox, Coord(100.0, 100,0.0), "max", "Upper bound of the liquid box")),
    m_fluidDensity(initData(&m_fluidDensity, (Real)1.0f, "fluidDensity", "Fluid Density"))
{

}



template <class DataTypes>
BuoyantForceField<DataTypes>::~BuoyantForceField()
{

}



template <class DataTypes>
void BuoyantForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();
    m_topology = this->getContext()->getMeshTopology();

    if (m_fluidDensity.getValue() <= 0.f)
    {
        serr << "Warning(BuoyantForceField):The density of the fluid is negative!" << sendl;
    }
}



template <class DataTypes>
void BuoyantForceField<DataTypes>::addForce(DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v, const core::MechanicalParams* /* mparams */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    if (m_topology)
    {
        int nbTetrahedra = m_topology->getNbTetrahedra();

//		    m_topology->getTetra(5);

        if (!nbTetrahedra)
        {
            serr << "Error(BuoyantForceField):No tetrahedron found in the topology" << sendl;
        }
        else
        {

            Real wholeVolume = static_cast<Real>(0.0f);
            Deriv gravity = this->getContext()->getLocalGravity();
            for (int i = 0 ; i < m_topology->getNbTetras() ; i++)
            {
                Tetra tetra = m_topology->getTetra(i);

                if ( isTetraInFluid(tetra, x))
                {
                    Deriv ab = x[tetra[1]] - x[tetra[0]];
                    Deriv ac = x[tetra[2]] - x[tetra[0]];
                    Deriv ad = x[tetra[3]] - x[tetra[0]];
                    Real volume = fabs(dot(ab.cross(ac), ad)) / static_cast<Real>(6.0);
                    wholeVolume += volume;
//
                    Real intensity = - m_fluidDensity.getValue() * volume / static_cast<Real>(4.0);
                    Deriv force = gravity * intensity ;

                    for ( int j = 0 ; j < 4 ; j++)
                    {
                        f[tetra[j]] += force;
                    }
                }
            }
//                   Real intensity = - m_fluidDensity.getValue() * wholeVolume / ( static_cast<Real>(4.0) * static_cast<Real>(m_topology->getNbPoints()));
//                   Deriv force = gravity * intensity ;
//
//                   for (int i = 0 ; i < m_topology->getNbTetras() ; i++)
//                   {
//                       Tetra tetra = m_topology->getTetra(i);
//                        for ( int j = 0 ; j < 4 ; j++)
//                        {
//                            f[tetra[j]] += force;
//                        }
//                   }

        }


    }
//
    d_f.endEdit();
}

template <class DataTypes>
bool BuoyantForceField<DataTypes>::isPointInFluid(const Coord &x) const
{
    if ( (m_maxBox == Coord()) && (m_minBox == Coord()) )
        return true;

    return ( (x[0] >= m_minBox.getValue()[0])
            && (x[0] <= m_maxBox.getValue()[0])
            && (x[1] >= m_minBox.getValue()[1])
            && (x[1] <= m_maxBox.getValue()[1])
            && (x[2] >= m_minBox.getValue()[2])
            && (x[2] <= m_maxBox.getValue()[2]) );
    return false;
}

template <class DataTypes>
bool BuoyantForceField<DataTypes>::isTetraInFluid(const Tetra &tetra, const VecCoord& x) const
{
    bool isAPointInFluid = false;

    for (unsigned int i = 0 ; i < 4 ; i++)
    {
        if (isPointInFluid(x[ tetra[i] ] )  )
        {
            isAPointInFluid = true;
            break;
        }
    }

    return isAPointInFluid;
}


template<class DataTypes>
void BuoyantForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    glDisable(GL_LIGHTING);

    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.f, 0.f, 1.0f, 0.1f);

    glBegin(GL_QUADS);

    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);

    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);

    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);

    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);

    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);

    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);
    glEnd();

    glColor4f(0.f, 1.f, 1.0f, 1.f);

    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_minBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_minBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_minBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glVertex3d(m_maxBox.getValue()[0],m_maxBox.getValue()[1],m_maxBox.getValue()[2]);
    glEnd();

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL
