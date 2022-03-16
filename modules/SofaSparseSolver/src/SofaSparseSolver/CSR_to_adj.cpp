#include <SofaSparseSolver/CSR_to_adj.h>

namespace sofa::component::linearsolver
{
void CSR_to_adj(int n,int * M_colptr,int * M_rowind, type::vector<int>& adj, type::vector<int>& xadj , type::vector<int>& t_adj , type::vector<int>& t_xadj, type::vector<int>& tran_countvec)
{
    //Compute transpose in tran_colptr, tran_rowind, tran_values, tran_D
    tran_countvec.clear();
    tran_countvec.resize(n);

    //First we count the number of value on each row.
    for (int j=0;j<n;j++) {
        for (int i=M_colptr[j];i<M_colptr[j+1];i++) {
        int col = M_rowind[i];
        if (col>j) tran_countvec[col]++;
        }
    }

    //Now we make a scan to build tran_colptr
    t_xadj.resize(n+1);
    t_xadj[0] = 0;
    for (int j=0;j<n;j++) t_xadj[j+1] = t_xadj[j] + tran_countvec[j];

    //we clear tran_countvec because we use it now to store hown many values are written on each line
    tran_countvec.clear();
    tran_countvec.resize(n);

    t_adj.resize(t_xadj[n]);
    for (int j=0;j<n;j++) {
        for (int i=M_colptr[j];i<M_colptr[j+1];i++) {
            int line = M_rowind[i];
            if (line>j) {
                t_adj[t_xadj[line] + tran_countvec[line]] = j;
                tran_countvec[line]++;
        }
    }
    }

    adj.clear();
    xadj.resize(n+1);
    xadj[0] = 0;
    for (int j=0; j<n; j++)
    {
        //copy the lower part
        for (int ip = t_xadj[j]; ip < t_xadj[j+1]; ip++) {
            adj.push_back(t_adj[ip]);
        }

        //copy only the upper part
        for (int ip = M_colptr[j]; ip < M_colptr[j+1]; ip++) {
            int col = M_rowind[ip];
            if (col > j) adj.push_back(col);
        }

        xadj[j+1] = adj.size();
    }
}

}
