/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/core/objectmodel/AspectPool.h>
#include <gtest/gtest.h>

using namespace sofa::core::objectmodel;
using sofa::core::SOFA_DATA_MAX_ASPECTS;

TEST(AspectPoolTest, allocate)
{
    AspectPool p;
    {
        AspectRef refs[SOFA_DATA_MAX_ASPECTS];
        // allocate all available aspects
        for(int i = 0; i < SOFA_DATA_MAX_ASPECTS; ++i)
        {
            refs[i] = p.allocate();
            EXPECT_TRUE(refs[i] != 0);
            EXPECT_EQ(refs[i]->aspectID(), i);
        }
        AspectRef extraRef = p.allocate(); // try to allocate one more aspect
        EXPECT_TRUE(extraRef == 0); // it should have returned a null pointer

        refs[SOFA_DATA_MAX_ASPECTS/2].reset(); // release an aspect
        extraRef = p.allocate(); // allocate a new one: it should return the ID of the previously released one.
        EXPECT_TRUE(extraRef != 0);
        EXPECT_EQ(extraRef->aspectID(), SOFA_DATA_MAX_ASPECTS/2);
    }

    AspectRef newRef = p.allocate(); // allocate an aspect after all have been released.
    EXPECT_TRUE(newRef != 0);
    EXPECT_EQ(newRef->aspectID(), SOFA_DATA_MAX_ASPECTS/2); // it should return the 0 aspect

    if(SOFA_DATA_MAX_ASPECTS > 1)
    {
        AspectRef extraRef = p.allocate();
        EXPECT_TRUE(extraRef != 0);
        EXPECT_EQ(extraRef->aspectID(), SOFA_DATA_MAX_ASPECTS-1); // it should return the last aspect
    }
}

