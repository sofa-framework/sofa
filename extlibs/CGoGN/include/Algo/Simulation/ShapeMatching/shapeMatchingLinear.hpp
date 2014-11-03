/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Simulation
{

namespace ShapeMatching
{

template <typename PFP>
void ShapeMatchingLinear<PFP>::computeAqqMatrix()
{
    //A_{qq} = (Sum_i{ m_{i} q_{i} q_i^{T} })^{-1}

    m_aqq = Eigen::Matrix3d::Zero();

    for(unsigned int i = 0 ; i < this->m_q.size() ; ++i)
    {
        m_aqq(0,0) += this->m_q[i][0] * this->m_q[i][0];
        m_aqq(1,0) += this->m_q[i][1] * this->m_q[i][0];
        m_aqq(2,0) += this->m_q[i][2] * this->m_q[i][0];

        m_aqq(0,1) += this->m_q[i][0] * this->m_q[i][1];
        m_aqq(1,1) += this->m_q[i][1] * this->m_q[i][1];
        m_aqq(2,1) += this->m_q[i][2] * this->m_q[i][1];

        m_aqq(0,2) += this->m_q[i][0] * this->m_q[i][2];
        m_aqq(1,2) += this->m_q[i][1] * this->m_q[i][2];
        m_aqq(2,2) += this->m_q[i][2] * this->m_q[i][2];
    }

    m_aqq = m_aqq.inverse();
}


template <typename PFP>
void ShapeMatchingLinear<PFP>::initialize()
{
    this->ShapeMatching<PFP>::initialize();

    computeAqqMatrix();
}

template <typename PFP>
void ShapeMatchingLinear<PFP>::shapeMatch()
{
    // p_{i}
    std::vector<Eigen::Vector3d> m_p;

    m_p.reserve(this->m_position.nbElements());

    //1.
    Eigen::Vector3d xcm = this->massCenter();

    for(unsigned int i = this->m_position.begin() ; i < this->m_position.end() ; this->m_position.next(i))
    {
       //this->m_p[i] = VEC3(this->m_position[i] - xcm); //p_{i} = x_{i} - x_{cm}

        Eigen::Vector3d tmp ;
        for (unsigned int j = 0 ; j < 3 ; ++j)
            tmp(j) = this->m_position[i][j] ;

       m_p[i] = tmp - xcm; //p_{i} = x_{i} - x_{cm}
    }

    //2.
    Eigen::Matrix3d apq = Eigen::Matrix3d::Zero();

    for(unsigned int i=0 ; i < m_p.size() ; ++i)
    {
        apq(0,0) += m_p[i][0] * this->m_q[i][0];
        apq(0,1) += m_p[i][0] * this->m_q[i][1];
        apq(0,2) += m_p[i][0] * this->m_q[i][2];

        apq(1,0) += m_p[i][1] * this->m_q[i][0];
        apq(1,1) += m_p[i][1] * this->m_q[i][1];
        apq(1,2) += m_p[i][1] * this->m_q[i][2];

        apq(2,0) += m_p[i][2] * this->m_q[i][0];
        apq(2,1) += m_p[i][2] * this->m_q[i][1];
        apq(2,2) += m_p[i][2] * this->m_q[i][2];
    }

    Eigen::Matrix3d S = apq.transpose() * apq ; //symmetric matrix

    //3. Jacobi Diagonalisation
    Eigen::EigenSolver<Eigen::Matrix3d> es(S);

    //V * D * V^(-1)
    Eigen::Matrix3d D = es.pseudoEigenvalueMatrix();
    Eigen::Matrix3d U = es.pseudoEigenvectors() ;

    for(int j = 0; j < 3; j++)
    {
        if(D(j,j) <= 0)
        {
            D(j,j) = 0.05f;
        }
        D(j,j) = 1.0f/sqrt(D(j,j));
    }

    S = U * D * U.transpose();


    Eigen::Matrix3d R = apq * S; //S^{-1}

    //4.
    Eigen::Matrix3d A = apq * m_aqq; //

    REAL det = A.determinant();
    det = 1.0f/powf(det,1.0f/3.0f);

    // \beta * A + (1 - \beta) * R
    if(std::isfinite(det))
    {
        R(0,0) += (m_beta * A(0,0) * det) + ((1.0f - m_beta) * R(0,0));
        R(0,1) += (m_beta * A(0,1) * det) + ((1.0f - m_beta) * R(0,1));
        R(0,2) += (m_beta * A(0,2) * det) + ((1.0f - m_beta) * R(0,2));

        R(1,0) += (m_beta * A(1,0) * det) + ((1.0f - m_beta) * R(1,0));
        R(1,1) += (m_beta * A(1,1) * det) + ((1.0f - m_beta) * R(1,1));
        R(1,2) += (m_beta * A(1,2) * det) + ((1.0f - m_beta) * R(1,2));

        R(2,0) += (m_beta * A(2,0) * det) + ((1.0f - m_beta) * R(2,0));
        R(2,1) += (m_beta * A(2,1) * det) + ((1.0f - m_beta) * R(2,1));
        R(2,2) += (m_beta * A(2,2) * det) + ((1.0f - m_beta) * R(2,2));
    }
    else
    {
        //no linear deformation (planar config ?), applying identity
        R = Eigen::Matrix3d::Identity();
    }

    //5.
    for(unsigned int i = this->m_goal.begin() ; i < this->m_goal.end() ; this->m_goal.next(i))
    {
        //this->m_goal[i] = R * this->m_q[i] + xcm; // g_{i} = R * q_i + x_{cm}

        Eigen::Vector3d tmp = R * this->m_q[i] + xcm; // g_{i} = R * q_i + x_{cm}

        VEC3 g;
        for (unsigned int j = 0 ; j < 3 ; ++j)
             g[j] = tmp(j);

         this->m_goal[i] = g;
    }

}

} // namespace ShapeMatching

} // namespace Simulation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
