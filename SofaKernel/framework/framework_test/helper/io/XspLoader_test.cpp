/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

#include <sofa/helper/system/FileRepository.h>

#include <sofa/helper/io/XspLoader.h>
using sofa::helper::io::XspLoader;
using sofa::helper::io::XspLoaderDataHook;

#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3;

namespace
{

class XspLoader_test : public BaseTest
{
protected:
    class XspData : public XspLoaderDataHook
    {
    public:
        size_t m_numMasses {0};
        size_t m_numSprings {0};
        std::vector<Vec3> m_masses;
        std::vector<Vec3> m_xtra;
        std::vector<std::tuple<int,int>> m_springs;
        bool m_hasGravity {false};
        bool m_hasViscosity {false};
        ~XspData() override {}
        void setNumMasses(size_t n) override  { m_numMasses = n; }
        void setNumSprings(size_t n) override  { m_numSprings = n; }
        void addMass(SReal px, SReal py, SReal pz, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) override
        {
            m_masses.push_back(Vec3(px,py,pz));
        }
        void addSpring(size_t m1, size_t m2, SReal /*ks*/, SReal /*kd*/, SReal /*initpos*/) override
        {
            m_springs.push_back(std::tuple<int,int>(m1,m2));
        }
        void addVectorSpring(size_t m1, size_t m2, SReal ks, SReal kd, SReal initpos, SReal restx, SReal resty, SReal restz) override
        {
            addSpring(m1, m2, ks, kd, initpos);
            m_xtra.push_back(Vec3(restx, resty, restz));
        }
        void setGravity(SReal /*gx*/, SReal /*gy*/, SReal /*gz*/) override
        {
            EXPECT_FALSE(m_hasGravity) << "Duplicated call to setGravity";
            m_hasGravity = true;
        }
        void setViscosity(SReal /*visc*/) override
        {
            EXPECT_FALSE(m_hasViscosity) << "Duplicated call to setViscosity";
            m_hasViscosity = true;
        }
    };

    void SetUp()
    {
        sofa::helper::system::DataRepository.addFirstPath(FRAMEWORK_TEST_RESOURCES_DIR);
    }
    void TearDown()
    {
        sofa::helper::system::DataRepository.removePath(FRAMEWORK_TEST_RESOURCES_DIR);
    }

    void loadFile(const std::string& filename, bool hasXtra)
    {
        XspData data;
        auto filePath = sofa::helper::system::DataRepository.getFile(filename);
        XspLoader::Load(filePath, data);

        ASSERT_EQ(data.m_numMasses, data.m_masses.size()) << "Number of 'masses' mismatch";
        ASSERT_EQ(data.m_numSprings, data.m_springs.size()) << "Number of 'springs' mismatch";
        ASSERT_EQ(data.m_numMasses, size_t(6)) << "Wrong number of 'masses'";
        ASSERT_EQ(data.m_numSprings, size_t(5)) << "Wrong number of 'springs'";
        for(unsigned int i=0;i<5;i++)
        {
            ASSERT_EQ(std::get<0>(data.m_springs[i]), i+0);
            ASSERT_EQ(std::get<1>(data.m_springs[i]), i+1);
        }
        EXPECT_FALSE(data.m_hasGravity);
        EXPECT_FALSE(data.m_hasViscosity);

        /// This is the case when handling Xsp 4.0 format.
        if(hasXtra)
        {
            EXPECT_EQ(data.m_numSprings, data.m_xtra.size()) << "Number of 'springs' mismatch with extra params";

            double va = 1.0;
            for(unsigned int i=0;i<5;i++)
            {
                ASSERT_NEAR(data.m_xtra[i].x(), 0.1+va, 0.005);
                ASSERT_NEAR(data.m_xtra[i].y(), 0.2+va, 0.005);
                ASSERT_NEAR(data.m_xtra[i].z(), 0.3+va, 0.005);
                va+=1.0;
            }
        }
    }
};

TEST_F(XspLoader_test, loadXs3File)
{
    this->loadFile("mesh/test.xs3", false);
}

TEST_F(XspLoader_test, loadXs4File)
{
    this->loadFile("mesh/test.xs4", true);
}


}// namespace sofa
