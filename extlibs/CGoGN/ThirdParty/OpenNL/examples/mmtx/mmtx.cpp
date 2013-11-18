/*
 *  Copyright (c) 2004-2010, Bruno Levy
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact: Bruno Levy
 *
 *     levy@loria.fr
 *
 *     ALICE Project
 *     LORIA, INRIA Lorraine, 
 *     Campus Scientifique, BP 239
 *     54506 VANDOEUVRE LES NANCY CEDEX 
 *     FRANCE
 *
 */


#include <NL/nl.h> 

#include <vector>
#include <set>
#include <map> 
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

#include <math.h>
#include <assert.h>
#include "nb_coo.h"
        
typedef std::map<int,double> mapVect;
typedef std::map<int,mapVect> mapMatrix;


bool isSymmetric(mapMatrix & M) {
  
  for (mapMatrix::iterator it = M.begin(); it != M.end() ; ++it) {
      int i   = it->first;
      for (mapVect::iterator it2 = it->second.begin() ; it2!=it->second.end() ; ++it2) {
	   int j      = it2->first ;
           double val = it2->second ;
           if (!M.count(j)) {
	        std::cout << i  << " " << j << "but" << j << " not present"   << std::endl;
                return false;
           }
	   else if ((!M[j].count(i)) || (M[j][i]!=M[i][j])) {
	        if (M[j].count(i))
		std::cout << i  << " " << j << " "<< val << " " << M[j][i] << " " <<  M[j][i] <<std::endl;
                else std::cout << j << " " << i  << " non present " << "val  " << val << " " << M[i][j]  << std::endl;
		return false;
          }
      }	   
  }    
  return true;
}


int main(int argc,char * argv[])
{

  if (argc < 5) {
      std::cerr << "usage : " << argv[0] << " type_solver max_iter precision in.dat" << std::endl;
      return -1;
  } 
  coo_matrix<NLuint,NLdouble> coo = read_coo_matrix<NLuint,NLdouble> (const_cast<const char *>(argv[4]));
  std::cout << "rows = " << coo.num_cols << " cols = "  << coo.num_rows <<  " nnz = " << coo.num_nonzeros << std::endl;
  mapMatrix map_a;

  // insertion of triplet (I,(J,V)) in the multimap
  for(unsigned int i = 0; i < coo.num_nonzeros ; ++i) {
        map_a[coo.I[i]][coo.J[i]]=coo.V[i];
  } 
  int max_iter = atoi(argv[2]);
  double epsilon = atof(argv[3]);
  const char * type_solver =argv[1];

  nlNewContext() ;
  if (!strcmp(type_solver,"CG")) {
            nlSolverParameteri(NL_SOLVER, NL_CG) ;
            nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
        }
	else if (!strcmp(type_solver,"BICGSTAB")) {
            nlSolverParameteri(NL_SOLVER, NL_BICGSTAB) ;
            nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
        }
	else if (!strcmp(type_solver,"GMRES")) {
            nlSolverParameteri(NL_SOLVER, NL_GMRES) ;
        }
    else if (!strcmp(type_solver,"SUPERLU")) {
        if(nlInitExtension("SUPERLU")) {
            nlSolverParameteri(NL_SOLVER, NL_PERM_SUPERLU_EXT) ;
        } else {
            std::cerr << "OpenNL has not been compiled with SuperLU support." << std::endl;
            exit(-1);
        }
    }else if(
        !strcmp(type_solver,"FLOAT_CRS") 
        || !strcmp(type_solver,"DOUBLE_CRS")
        || !strcmp(type_solver,"FLOAT_BCRS2")
        || !strcmp(type_solver,"DOUBLE_BCRS2")
        || !strcmp(type_solver,"FLOAT_ELL")
        || !strcmp(type_solver,"DOUBLE_ELL")
        || !strcmp(type_solver,"FLOAT_HYB")
        || !strcmp(type_solver,"DOUBLE_HYB")
        ) {
        if(nlInitExtension("CNC")) {
  	        if (!strcmp(type_solver,"FLOAT_CRS")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_FLOAT_CRS) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }
            else if (!strcmp(type_solver,"DOUBLE_CRS")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_DOUBLE_CRS) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }
            else if (!strcmp(type_solver,"FLOAT_BCRS2")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_FLOAT_BCRS2) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }
            else if (!strcmp(type_solver,"DOUBLE_BCRS2")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_DOUBLE_BCRS2) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }
            else if (!strcmp(type_solver,"FLOAT_ELL")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_FLOAT_ELL) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            } 
	        else if (!strcmp(type_solver,"DOUBLE_ELL")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_DOUBLE_ELL) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }
	        else if (!strcmp(type_solver,"FLOAT_HYB")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_FLOAT_HYB) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }
	        else if (!strcmp(type_solver,"DOUBLE_HYB")) {
                nlSolverParameteri(NL_SOLVER, NL_CNC_DOUBLE_HYB) ;
                nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI) ;
            }

        } else {
            std::cerr << "OpenNL has not been compiled with CNC support." << std::endl;  
            exit(-1);
        }
    } else {
        std::cerr << "type_solver must belong to { CG | BICGSTAB | GMRES | "
                  << "SUPERLU | FLOAT_CRS | FLOAT_BCRS2 | DOUBLE_CRS | "
                  << "DOUBLE_BCRS2 | FLOAT_ELL | DOUBLE_ELL | FLOAT_HYB |"
                  << "DOUBLE_HYB } "
	              << std::endl;
	    exit(-1);
	}
  
  nlSolverParameteri(NL_NB_VARIABLES, coo.num_cols) ;
  if (coo.num_cols != coo.num_rows) {
      std::cout << "the matrix ist not square : using least-squares solutions" << std::endl;
      nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE) ;
  }
  else if (!isSymmetric(map_a)) {
      std::cout << "the matrix is not symmetric : using least-squares solutions" << std::endl;
      nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE) ;
  }
  else
      nlSolverParameteri(NL_LEAST_SQUARES, NL_FALSE);      
  nlSolverParameteri(NL_MAX_ITERATIONS, max_iter) ;
  nlSolverParameterd(NL_THRESHOLD, epsilon) ;
  nlBegin(NL_SYSTEM) ;
  nlBegin(NL_MATRIX) ;

  // generation d'un b
  std::vector<NLdouble> b(coo.num_rows,0.);
  for(NLuint i = 0; i < coo.num_rows; i++){
          b[i] = (double)(rand() / (RAND_MAX + 1.0));
  }

  //NLdouble scale_factor = 1.0 / (coo.num_rows*coo.num_cols);
  for(unsigned int i =0 ; i < coo.num_rows ; ++i) {
       nlRowParameterd(NL_RIGHT_HAND_SIDE,b[i]);
       //nlRowParameterd(NL_ROW_SCALING,scale_factor);
       // starting row i 
       nlBegin(NL_ROW);
       if (map_a.count(i)) {
           // we go trough the i-th row
	   for (mapVect::iterator it=map_a[i].begin(); it!=map_a[i].end(); ++it)
                nlCoefficient(it->first,it->second);
       }
       nlEnd(NL_ROW) ;      
  }
  nlEnd(NL_MATRIX) ;
  nlEnd(NL_SYSTEM) ;
  std::cout << "Solving ..." << std::endl ;
  double time ;
  NLint iterations;
  nlSolve() ;
  nlGetDoublev(NL_ELAPSED_TIME, &time) ;
  nlGetIntergerv(NL_USED_ITERATIONS, &iterations); 
  std::cout << "Solver time: " << time << std::endl ;
  std::cout << "Used iterations: " << iterations << std::endl ;
  nlDeleteContext(nlGetCurrent()) ;
  return 0;
}

