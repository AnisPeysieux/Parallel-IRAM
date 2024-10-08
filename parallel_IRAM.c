#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include "mmio.h"
#include <float.h>

enum mat_layout {
  COLUMN_MAJOR,
  ROW_MAJOR
};

//Print matrix for debugging
void print_mat_format(double* mat, int n_rows, int n_cols, int precision, enum mat_layout layout, char* sep_elem, char* sep_part, char* begin_part, char* end_part, char* begin_mat, char* end_mat)
{
  //printf("%d rows, %d cols:\n", n_rows, n_cols);
  double (*mat_cast_rm)[n_cols] = (double(*) [n_cols]) mat;
  double (*mat_cast_cm)[n_rows] = (double(*) [n_rows]) mat;
  
  fprintf(stderr, "%s", begin_mat);
  for(int row = 0; row < n_rows; ++row)
  {
    if(row != 0)
    {
      fprintf(stderr, "%s", sep_part);
    }
    fprintf(stderr, "%s", begin_part);
    for(int col = 0; col < n_cols; ++col)
    {
      //printf(" %d,%d -> ", row, col);
      if(col == 0)
      {
          if(layout == ROW_MAJOR)
            fprintf(stderr, "%.*f", precision, mat_cast_rm[row][col]);
          else if(layout == COLUMN_MAJOR)
            fprintf(stderr, "%.*f", precision, mat_cast_cm[col][row]);
      }
      else
      {
          fprintf(stderr, "%s", sep_elem);
          if(layout == ROW_MAJOR)
            fprintf(stderr, "%.*f", precision, mat_cast_rm[row][col]);
          else if(layout == COLUMN_MAJOR)
            fprintf(stderr, "%.*f", precision, mat_cast_cm[col][row]);
      }
    }
    fprintf(stderr, "%s", end_part);
  }
  fprintf(stderr,"%s", end_mat);

}

void print_mat_format_parallel(MPI_Comm comm, double* mat, int n_rows, int n_cols, int precision, enum mat_layout layout, char* sep_elem, char* sep_part, char* begin_part, char* end_part, char* begin_mat, char* end_mat)
{
  //printf("%d rows, %d cols:\n", n_rows, n_cols);
  double (*mat_cast_rm)[n_cols] = (double(*) [n_cols]) mat;
  double (*mat_cast_cm)[n_rows] = (double(*) [n_rows]) mat;
  
  int my_rank, comm_size;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  for(int rank = 0; rank < comm_size; ++rank)
  {
    if(my_rank == rank)
    {
      fprintf(stderr, "rank %d:\n", my_rank);
      fprintf(stderr, "%s", begin_mat);
      for(int row = 0; row < n_rows; ++row)
      {
        if(row != 0)
        {
          fprintf(stderr, "%s", sep_part);
        }
        fprintf(stderr, "%s", begin_part);
        for(int col = 0; col < n_cols; ++col)
        {
          //printf(" %d,%d -> ", row, col);
          if(col == 0)
          {
              if(layout == ROW_MAJOR)
                fprintf(stderr, "%.*f", precision, mat_cast_rm[row][col]);
              else if(layout == COLUMN_MAJOR)
                fprintf(stderr, "%.*f", precision, mat_cast_cm[col][row]);
          }
          else
          {
              fprintf(stderr, "%s", sep_elem);
              if(layout == ROW_MAJOR)
                fprintf(stderr, "%.*f", precision, mat_cast_rm[row][col]);
              else if(layout == COLUMN_MAJOR)
                fprintf(stderr, "%.*f", precision, mat_cast_cm[col][row]);
          }
        }
        fprintf(stderr, "%s", end_part);
      }
      fprintf(stderr, "%s", end_mat);
      fflush(stderr);
    }
    MPI_Barrier(comm);
  }
}

void print_mat_format_parallel_int(MPI_Comm comm, int* mat, int n_rows, int n_cols, int precision, enum mat_layout layout, char* sep_elem, char* sep_part, char* begin_part, char* end_part, char* begin_mat, char* end_mat)
{
  //printf("%d rows, %d cols:\n", n_rows, n_cols);
  int (*mat_cast_rm)[n_cols] = (int(*) [n_cols]) mat;
  int (*mat_cast_cm)[n_rows] = (int(*) [n_rows]) mat;
  
  int my_rank, comm_size;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  for(int rank = 0; rank < comm_size; ++rank)
  {
    if(my_rank == rank)
    {
      fprintf(stderr, "rank %d:\n", my_rank);
      fprintf(stderr, "%s", begin_mat);
      for(int row = 0; row < n_rows; ++row)
      {
        if(row != 0)
        {
          fprintf(stderr, "%s", sep_part);
        }
        fprintf(stderr, "%s", begin_part);
        for(int col = 0; col < n_cols; ++col)
        {
          //printf(" %d,%d -> ", row, col);
          if(col == 0)
          {
              if(layout == ROW_MAJOR)
                fprintf(stderr, "%d", mat_cast_rm[row][col]);
              else if(layout == COLUMN_MAJOR)
                fprintf(stderr, "%d", mat_cast_cm[col][row]);
          }
          else
          {
              fprintf(stderr, "%s", sep_elem);
              if(layout == ROW_MAJOR)
                fprintf(stderr, "%d", mat_cast_rm[row][col]);
              else if(layout == COLUMN_MAJOR)
                fprintf(stderr, "%d", mat_cast_cm[col][row]);
          }
        }
        fprintf(stderr, "%s", end_part);
      }
      fprintf(stderr, "%s", end_mat);
      fflush(stderr);
    }
    MPI_Barrier(comm);
  }
}

enum mpi_tags {
  distribute_A_msg
};

//Repart matrix, allocate memory, ...
void init_work(MPI_Comm comm, int n_rows_A, int n_cols_A, int subspace_dim, double* global_A,
              int* n_local_rows_A, int* n_local_cols_A, int* n_local_rows_V,
              int* n_local_cols_V, int** n_rows_V_per_process, int* n_local_rows_H, int* n_local_cols_H, double** local_A, double** local_H, double** local_V)
{
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  *n_local_cols_A = n_cols_A/size;
  if(rank < n_cols_A % size)
  {
    (*n_local_cols_A)++;
  }
  *n_local_rows_A = n_rows_A;
  *n_local_rows_V = *n_local_cols_A;
  *n_local_cols_V = subspace_dim + 1;
  *n_local_rows_H = subspace_dim + 1;
  *n_local_cols_H = subspace_dim;

  *n_rows_V_per_process = malloc(size * sizeof(int));
  for(int i = 0; i < size; ++i)
  {
    (*n_rows_V_per_process)[i] = n_cols_A/size;
    if(i < n_cols_A % size)
    {
      (*n_rows_V_per_process)[i]++;
    }
  }
  *local_A = malloc((*n_local_rows_A) * (*n_local_cols_A) * sizeof(double));
  *local_V = malloc((*n_local_rows_V) * (*n_local_cols_V) * sizeof(double));
  *local_H = malloc((*n_local_rows_H) * (*n_local_cols_H) * sizeof(double));
  memset(*local_A, 0, (*n_local_rows_A) * (*n_local_cols_A) * sizeof(double));
  memset(*local_V, 0, (*n_local_rows_V) * (*n_local_cols_V) * sizeof(double));
  memset(*local_H, 0, (*n_local_rows_H) * (*n_local_cols_H) * sizeof(double));

  if(rank == 0)
  {
    double (*A)[n_cols_A] = (double(*)[n_cols_A]) global_A;
    int first_col_to_send = *n_local_cols_A;
    for(int dest_rank = 1; dest_rank < size; ++dest_rank)
    {
      int n_rows_to_send = n_rows_A;
      int n_cols_to_send = n_cols_A/size;
      if(dest_rank < n_cols_A % size)
      {
        n_cols_to_send++;
      }
     
      for(int row = 0; row < n_rows_to_send; ++row)
      {
        MPI_Ssend(&(A[row][first_col_to_send]), n_cols_to_send, MPI_DOUBLE, dest_rank, distribute_A_msg, comm);
      }

      //Prepare next iteration
      first_col_to_send += n_cols_to_send;
    }

    double (*A_0)[*n_local_cols_A] = (double(*)[*n_local_cols_A]) *local_A;
    for(int row = 0; row < *n_local_rows_A; ++row)
    {
      for(int col = 0; col < *n_local_cols_A; ++col)
      {
        A_0[row][col] = A[row][col];
      }
    }

  }
  else
  {
    double (*A)[*n_local_cols_A] = (double(*)[*n_local_cols_A]) *local_A;
    for(int row = 0; row < (*n_local_rows_A); ++row)
    {
      MPI_Recv(&(A[row][0]), *n_local_cols_A, MPI_DOUBLE, 0, distribute_A_msg, comm, MPI_STATUS_IGNORE);
    }
  }

  MPI_Barrier(comm);
}

//Basic matrix operations
void dot(double* v1, int v1_stride, double* v2, int v2_stride, int n, double* res, MPI_Comm comm)
{
  double local_res = 0;
  for(int i = 0; i < n; ++i)
  {
    local_res += v1[i*v1_stride] * v2[i*v2_stride];
  }
  MPI_Allreduce(&local_res, res, 1, MPI_DOUBLE, MPI_SUM, comm);
}

void daxpy(double* v1, int v1_stride, double* v2, int v2_stride, int n, double alpha)
{
  for(int i = 0; i < n; ++i)
  {
    v1[i*v1_stride] += alpha * v2[i*v2_stride];
  }
}

void mat_vect_prod(double* m, int n_rows_m, int m_first_dim, int n_cols_m, double* v, int v_stride, double* vres, int vres_stride, int vres_size, int* n_rows_V_per_process, MPI_Comm comm)
{
  //TODO: static ?
  int rank; MPI_Comm_rank(comm, &rank);
  double (*M)[m_first_dim] = (double(*)[n_cols_m]) m;
  double* local_res = malloc(n_rows_m * sizeof(double));
  double* my_res = malloc(vres_size * sizeof(double));
  memset(local_res, 0, n_rows_m * sizeof(double));
  memset(my_res, 0, vres_size * sizeof(double));
  for(int row = 0; row < n_rows_m; ++row)
  {
    local_res[row] = 0;
    for(int col = 0; col < n_cols_m; ++col)
    {
      local_res[row] += M[row][col] * v[col * v_stride];
    }

  }
  MPI_Reduce_scatter(local_res, my_res, n_rows_V_per_process, MPI_DOUBLE, MPI_SUM, comm);
  for(int row = 0; row < vres_size; ++row)
  {
      vres[row * vres_stride] = my_res[row];
  }
  free(local_res);
  free(my_res);
}

void norm2(double* v, int stride, int n, double* res, MPI_Comm comm)
{
  double local_res = 0;
  for(int i = 0; i < n; ++i)
  {
    local_res += fabs(v[i*stride]) * fabs(v[i*stride]);
  }
  MPI_Allreduce(&local_res, res, 1, MPI_DOUBLE, MPI_SUM, comm);
  *res = sqrt(*res);
}

void scale(double* v, int v_stride, int n, double alpha)
{
  for(int i = 0; i < n; ++i)
  {
    v[i*v_stride] *= alpha;
  }
}

//QR algorithm using LAPACK
void qr_algorithm_row_major(double* m_A, double* a_out, double* r_out, double* q_out, int dim, int iter)
{
  
  double* tau = malloc(dim * sizeof(double));
  double (*R)[dim] = (double(*)[dim]) r_out;
  double (*A_next)[dim] = (double(*)[dim]) a_out;
  double (*Q)[dim] = (double(*)[dim]) q_out;
  double (*A)[dim] = (double(*)[dim]) m_A;

  for(int row = 0; row < dim; ++row)
  {
    for(int col = 0; col < dim; ++col)
    {
      A_next[row][col] = A[row][col];
    }
  }

  for(int i = 0; i < iter; ++i)
  {
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dim, dim, (double*)A_next, dim, tau);
    for(int row = 0; row < dim; ++row)
    {
      for(int col = 0; col < dim; ++col)
      {
        if(col >= row)
        {
          R[row][col] = A_next[row][col];
        }
        else
        {
          R[row][col] = 0;
        }
      }
    }
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, dim, dim,
                             dim, (double*)A_next, dim,
                             tau);
    cblas_dcopy(dim * dim, (double*)A_next, 1, (double*)Q, 1);
    memset(A_next, 0, dim * dim * sizeof(double));
    cblas_dgemm ( CblasRowMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  dim, dim, dim,
                  1, (double*)R, dim,
                  (double*)Q, dim, 1,
                  (double*)A_next, dim);
  }

  free(tau);

}

//Arnoldi factorization using basic matrix operations defined before
void Arnoldi_Factorization(double* _A, int n_local_rows_A, int n_local_cols_A,
                            double* _V, int n_local_rows_V, int n_local_cols_V, int* n_rows_V_per_process,
                            double* _H, int n_local_rows_H, int n_local_cols_H,
                            int subspace_dim, int first_col, MPI_Comm comm)
{
  int rank; MPI_Comm_rank(comm, &rank);

  double (*A)[n_local_cols_A] = (double(*)[n_local_cols_A]) _A;
  double (*V)[n_local_cols_V] = (double(*)[n_local_cols_V]) _V;
  double (*H)[n_local_cols_H] = (double(*)[n_local_cols_H]) _H;
  double* first_vector = malloc(n_local_rows_V * sizeof(double));
  int first_vector_int = 0;
  for(int i = 0; i < rank; ++i)
  {
    first_vector_int += n_rows_V_per_process[i];
  }

  //Si on fait Arnoldi depuis le debut, on doit choisir un vecteur de demarrage
  if(first_col == 0)
  {
    for(int i = 0; i < n_local_rows_V; ++i)
    {
      first_vector[i] = i + first_vector_int + 1;
      //V[i][0] = (double)rand() / (double)RAND_MAX;
      V[i][0] = i + first_vector_int + 1;
    }
   // print_mat_format_parallel(comm, first_vector, 1, n_local_rows_V, 5, ROW_MAJOR, ", ", ",\n", "{", "}", "first_vector: {\n", "\n}\n");
    double v0_norm = 0;
    norm2(&(V[0][0]), n_local_cols_V, n_local_rows_V, &v0_norm, comm);
    for(int i = 0; i < n_local_rows_V; ++i)
    {
      V[i][0] /= v0_norm;
    }
  }
  //Si c'est un redemarrage, on utilise la derniere colonne calculee avant la premiere que l'on va calculer
  else
  {
    first_col--;
  }
  
  for(int k = first_col; k < subspace_dim; ++k)
  {
    mat_vect_prod((double*) A, n_local_rows_A, n_local_cols_A, n_local_cols_A, &(V[0][k]), n_local_cols_V, &(V[0][k+1]), n_local_cols_V, n_local_rows_V, n_rows_V_per_process, comm);
    for(int j = 0; j <= k; ++j)
    {
      double local_h;
      dot(&(V[0][j]), n_local_cols_V, &(V[0][k+1]), n_local_cols_V, n_local_rows_V, &H[j][k], comm);
      daxpy(&(V[0][k+1]), n_local_cols_V, &(V[0][j]), n_local_cols_V, n_local_rows_V, -H[j][k]);
    }
    double local_h;
    norm2(&(V[0][k+1]), n_local_cols_V, n_local_rows_V, &(H[k+1][k]), comm);
    scale(&(V[0][k+1]), n_local_cols_V, n_local_rows_V, 1./H[k+1][k]);
  }
  //if(rank == 0) print_mat_format(H, subspace_dim+1, subspace_dim, 5, ROW_MAJOR, ", ", ",\n", "{", "}", "H Fin Arnoldi: {\n", "\n}\n");
  //if(rank == 0) print_mat_format(V, n_local_rows_V, subspace_dim+1, 5, ROW_MAJOR, ", ", ",\n", "{", "}", "V Fin Arnoldi: {\n", "\n}\n");
 //   if(rank == 0) print_mat_format(H, subspace_dim, subspace_dim, 5, ROW_MAJOR, ", ", ",\n", "{", "}", "H Arnoldi: {\n", "\n}\n");
   // exit(0);

  free(first_vector);

}

int comp_double(const void* _a, const void* _b)
{
  double a = fabs(*((double*) _a));
  double b = fabs(*((double*) _b));

  if(a < b)
    return 1;
  else if(a > b)
    return -1;
  else
    return 0;
}

//QR factorization using LAPACK
void QR_factorization(double* m_A, double* r_out, double* q_out, int dim)
{
  double* tau = malloc(dim * sizeof(double));
  double (*R)[dim] = (double(*)[dim]) r_out;
  double (*A_next)[dim] = malloc(dim * dim * sizeof(double));
  double (*Q)[dim] = (double(*)[dim]) q_out;
  double (*A)[dim] = (double(*)[dim]) m_A;

  for(int row = 0; row < dim; ++row)
  {
    for(int col = 0; col < dim; ++col)
    {
      A_next[row][col] = A[row][col];
    }
  }

  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, dim, dim, (double*)A_next, dim, tau);
  for(int row = 0; row < dim; ++row)
  {
    for(int col = 0; col < dim; ++col)
    {
      if(col >= row)
      {
        R[row][col] = A_next[row][col];
      }
      else
    {
        R[row][col] = 0;
      }
    }
  }
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, dim, dim,
                           dim, (double*)A_next, dim,
                           tau);
  cblas_dcopy(dim * dim, (double*)A_next, 1, (double*)Q, 1);

  free(tau);
  free(A_next);
}

void mat_mat_prod(double* m1, int n_rows_m1, int n_cols_m1, int m1_first_dim, int transpose_m1,  double* m2, int n_rows_m2, int n_cols_m2, int m2_first_dim, int transpose_m2, double* mres, int n_rows_mres, int n_cols_mres, int mres_first_dim)
{
  double (*M1)[m1_first_dim] = (double(*)[m1_first_dim]) m1;
  double (*M2)[m2_first_dim] = (double(*)[m2_first_dim]) m2;
  double (*MRES)[mres_first_dim] = (double(*)[mres_first_dim]) mres;

  if(transpose_m1 && transpose_m2)
  {
    #pragma omp parallel for
    for(int row_m1 = 0; row_m1 < n_rows_m1; ++row_m1)
    {
      for(int col_m2 = 0; col_m2 < n_cols_m2; ++col_m2)
      {
        MRES[row_m1][col_m2] = 0;
        for(int row_m2 = 0; row_m2 < n_rows_m2; ++row_m2)
        {
          double m1_elem;
          double m2_elem;
          m1_elem = M1[row_m2][row_m1];
          m2_elem = M2[col_m2][row_m2];
          MRES[row_m1][col_m2] +=  m1_elem * m2_elem;
        }
      }
    }

  }
  else if(transpose_m1 && !transpose_m2)
  {
    #pragma omp parallel for
    for(int row_m1 = 0; row_m1 < n_rows_m1; ++row_m1)
    {
      for(int col_m2 = 0; col_m2 < n_cols_m2; ++col_m2)
      {
        MRES[row_m1][col_m2] = 0;
        for(int row_m2 = 0; row_m2 < n_rows_m2; ++row_m2)
        {
          double m1_elem;
          double m2_elem;
          m1_elem = M1[row_m2][row_m1];
          m2_elem = M2[row_m2][col_m2];
          MRES[row_m1][col_m2] +=  m1_elem * m2_elem;
        }
      }
    }

  }
  else if(!transpose_m1 && transpose_m2)
  {
    #pragma omp parallel for
    for(int row_m1 = 0; row_m1 < n_rows_m1; ++row_m1)
    {
      for(int col_m2 = 0; col_m2 < n_cols_m2; ++col_m2)
      {
        MRES[row_m1][col_m2] = 0;
        for(int row_m2 = 0; row_m2 < n_rows_m2; ++row_m2)
        {
          double m1_elem;
          double m2_elem;
          m1_elem = M1[row_m1][row_m2];
          m2_elem = M2[col_m2][row_m2];
          MRES[row_m1][col_m2] +=  m1_elem * m2_elem;
        }
      }
    }

  }
  else
  {
    #pragma omp parallel for
    for(int row_m1 = 0; row_m1 < n_rows_m1; ++row_m1)
    {
      for(int col_m2 = 0; col_m2 < n_cols_m2; ++col_m2)
      {
        MRES[row_m1][col_m2] = 0;
        for(int row_m2 = 0; row_m2 < n_rows_m2; ++row_m2)
        {
          double m1_elem;
          double m2_elem;
          m1_elem = M1[row_m1][row_m2];
          m2_elem = M2[row_m2][col_m2];
          MRES[row_m1][col_m2] +=  m1_elem * m2_elem;
        }
      }
    }

  }

}

void sort_eigen_values(double* re, double* im, int n)
{
  for(int i = n - 1; i > 0; --i)
  {
    for(int j = 0; j < i; ++j)
    {
      //fprintf(stderr, "cmp [%d]=%f - [%d]=%f\n", j+1, re[j+1], j, re[j]);
      if(fabs(re[j+1]) > fabs(re[j]))
      {
        double tmp = re[j+1];
        //fprintf(stderr, "\tre[%d] <- = %f\n", j+1, re[j]);
        re[j+1] = re[j];
        //fprintf(stderr, "\tre[%d] <- = %f\n", j, tmp);
        re[j] = tmp;
        
        tmp = im[j+1];
        im[j+1] = im[j];
        im[j] = tmp;
      }
    }
  }
}

void IRAM(double* _A, int n_local_rows_A, int n_local_cols_A,
                            double* _V, int n_local_rows_V, int n_local_cols_V, int* n_rows_V_per_process,
                            double* _H, int n_local_rows_H, int n_local_cols_H,
                            int subspace_dim, int max_iter, int desired, double* residuals, int* n_iter, int qr_iter, double residual_max, double* eigen_values_r, double* eigen_values_i, double* eigen_vectors, MPI_Comm comm)
{

  int rank; MPI_Comm_rank(comm, &rank);
  double (*A)[n_local_cols_A] = (double(*)[n_local_cols_A]) _A;
  double (*V)[n_local_cols_V] = (double(*)[n_local_cols_V]) _V;
  double (*H)[n_local_cols_H] = (double(*)[n_local_cols_H]) _H;
  double (*HQ)[subspace_dim] = (double(*)[subspace_dim]) malloc(subspace_dim * subspace_dim * sizeof(double));
  double *H_cpy =  malloc(subspace_dim * subspace_dim * sizeof(double));
  double (*VQ)[subspace_dim] = (double(*)[n_local_cols_V]) malloc(n_local_rows_V * n_local_cols_V * sizeof(double));
  double (*tmp_V)[subspace_dim];
  double (*H_out)[subspace_dim] = malloc((subspace_dim) * (subspace_dim) * sizeof(double));
  double (*R_out)[subspace_dim] = malloc((subspace_dim) * (subspace_dim) * sizeof(double));
  double (*Q_out)[subspace_dim] = malloc((subspace_dim) * (subspace_dim) * sizeof(double));
  double (*Q)[subspace_dim] = malloc((subspace_dim) * (subspace_dim) * sizeof(double));
  double (*R)[subspace_dim] = malloc((subspace_dim) * (subspace_dim) * sizeof(double));
  double (*shifted_H)[subspace_dim] = malloc(subspace_dim * subspace_dim * sizeof(double));
  int* ifailr = malloc(subspace_dim * sizeof(int));
  double* eigen_values_r_2 = malloc(subspace_dim * sizeof(double*));
  double* eigen_values_i_2 = malloc(subspace_dim * sizeof(double*));
  lapack_logical* select = malloc(subspace_dim * sizeof(lapack_logical));
  memset(HQ, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(H_cpy, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(VQ, 0, n_local_rows_V * n_local_cols_V * sizeof(double));
  memset(H_out, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(R_out, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(Q_out, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(Q, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(R, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(shifted_H, 0, subspace_dim * subspace_dim * sizeof(double));
  memset(ifailr, 0, subspace_dim * sizeof(int));
  memset(eigen_values_r, 0, subspace_dim * sizeof(double));
  memset(eigen_values_i, 0, subspace_dim * sizeof(double));
  memset(select, 0, subspace_dim * sizeof(lapack_logical));
  
  int undesired = subspace_dim - desired;
  int converge;
  for(int i = 0; i < subspace_dim; ++i)
  {
    select[i] = 1;
  }

  for(int it = 0; it < max_iter; ++it)
  {
    *n_iter = it;

    //If we use QR algorithm
    if(qr_iter > 0)
    {
      //Compute eigenvalues and eigenvectors of H
      qr_algorithm_row_major((double*) H,(double*) H_out,(double*) R_out,(double*) Q_out, subspace_dim, qr_iter);
      for(int i = 0; i < subspace_dim; ++i)
      {
        eigen_values_r[i] = H_out[i][i];
      }
      qsort(eigen_values_r, subspace_dim, sizeof(double), comp_double);

      //Rank 0 compute residual
      if(rank == 0)
      {
        double max = fabs(Q_out[subspace_dim-1][0]);
        for(int i = 0; i < subspace_dim; ++i)
        {
          if(fabs(Q_out[subspace_dim-1][i]) > max)
          {
            max = fabs(Q_out[subspace_dim-1][i]);
          }
        }
        residuals[it] = max * H[subspace_dim][subspace_dim-1];
      }

      //Rank 0 broadcast residual to all processes
      MPI_Bcast(&(residuals[it]), 1, MPI_DOUBLE, 0, comm);

      if(residuals[it] < residual_max)
      {
        return;
      }

    }
    //If we use Lapack
    else
    {
      //Compute eigenvalues and eigenvectors of H
      memcpy(H_cpy, H, subspace_dim * subspace_dim * sizeof(double));
      LAPACKE_dhseqr(LAPACK_ROW_MAJOR, 'E', 'N', subspace_dim, 1, subspace_dim, (double*) H_cpy, subspace_dim, eigen_values_r, eigen_values_i, NULL, subspace_dim);
      sort_eigen_values(eigen_values_r, eigen_values_i, subspace_dim);
      memcpy(eigen_values_r_2, eigen_values_r, subspace_dim * sizeof(double));
      memcpy(eigen_values_i_2, eigen_values_i, subspace_dim * sizeof(double));
      int col_max;
      lapack_int info = LAPACKE_dhsein(LAPACK_ROW_MAJOR, 'R', 'Q', 'N', select, subspace_dim, (double*)H_cpy, subspace_dim, eigen_values_r_2, eigen_values_i_2, NULL, subspace_dim, eigen_vectors, subspace_dim, subspace_dim, &col_max, NULL, ifailr);

      //Rank 0 compute residual
      if(rank == 0)
      {
        double max = ((double(*)[subspace_dim])eigen_vectors)[subspace_dim-1][0];
        if(col_max > subspace_dim)
        {
          col_max = subspace_dim;
        }
        for(int col = 0; col < col_max; ++col)
        {
          if(fabs(((double(*)[subspace_dim])eigen_vectors)[subspace_dim-1][col]) > max)
          {
            max = fabs(((double(*)[subspace_dim])eigen_vectors)[subspace_dim-1][col]);
          }
        }
        residuals[it] = max * H[subspace_dim][subspace_dim-1];
      }

      //Rank 0 broadcast residual to all processes
      MPI_Bcast(&(residuals[it]), 1, MPI_DOUBLE, 0, comm);
      //if(rank == 0) fprintf(stderr, "diff = %.55f - %.55f = %.55f < %.55f\n", residuals[it], residual_max, residuals[it] - residual_max, DBL_EPSILON);
      if(residuals[it] < residual_max)
      {
        return;
      }
    }
    
    //Transform H and V for each shift
    for(int j = desired; j < subspace_dim; ++j)
    {
      //Apply shift
      for(int row = 0; row < subspace_dim; ++row)
      {
        for(int col = 0; col < subspace_dim; ++col)
        {
          if(row == col)
          {
            shifted_H[row][col] = H[row][col] - eigen_values_r[j];
          }
          else
          {
            shifted_H[row][col] = H[row][col];
          }
        }
      }

      QR_factorization((double*) shifted_H, (double*) R, (double*) Q, subspace_dim);
      
      mat_mat_prod((double*)H, subspace_dim, subspace_dim, subspace_dim, 0, (double*)Q, subspace_dim, subspace_dim, subspace_dim, 0, (double*)HQ, subspace_dim, subspace_dim, subspace_dim);
      mat_mat_prod((double*)Q, subspace_dim, subspace_dim, subspace_dim, 1, (double*)HQ, subspace_dim, subspace_dim, subspace_dim, 0, (double*)H, subspace_dim, subspace_dim, subspace_dim);
      mat_mat_prod((double*)V, n_local_rows_V, subspace_dim, n_local_cols_V, 0, (double*)Q, subspace_dim, subspace_dim, subspace_dim, 0, (double*)VQ, n_local_rows_V, subspace_dim, n_local_cols_V);
     
      tmp_V = VQ;
      VQ = V;
      V = tmp_V;
    }

    //New Arnoldi Factorization
    Arnoldi_Factorization((double*) A, n_local_rows_A, n_local_cols_A,
                          (double*) V, n_local_rows_V, n_local_cols_V, n_rows_V_per_process,
                          (double*) H, n_local_rows_H, n_local_cols_H,
                          subspace_dim, desired, MPI_COMM_WORLD);
    if(qr_iter > 0)
    {

      memcpy(eigen_vectors, Q_out, subspace_dim * subspace_dim * sizeof(double));
    }



  }


  free(HQ);
  free(H_cpy);
  free(VQ);
  free(H_out);
  free(R_out);
  free(Q_out);
  free(Q);
  free(R);
  free(shifted_H);
  free(ifailr);
  free(eigen_values_r_2);
  free(eigen_values_i_2);

}

void load_mat(double** ptrMat, int* nRow, int* nCol, FILE* file)
{
  int retCode;
  MM_typecode matcode;
  int nz;
  double* A = NULL;

  if (mm_read_banner(file, &matcode) != 0)
  {
    printf("! ERREUR ! Impossible de lire l'en-tête du fichier !\n");
    exit(EXIT_FAILURE);
  }
  if(mm_is_complex(matcode) != 0 && mm_is_matrix(matcode) != 0 && mm_is_sparse(matcode) != 0)
  {
    printf("! ERREUR ! Cette application ne reconnaît pas le format : %s\n", mm_typecode_to_str(matcode));
    exit(EXIT_FAILURE);
  }

  retCode = mm_read_mtx_crd_size(file, nRow, nCol, &nz); // Récupération des tailles
  if (retCode != 0)
  {
    printf("! ERREUR ! Tailles non reconnues !\n");
    exit(EXIT_FAILURE);
  }
  //A = (double*)mkl_malloc((*nRow)*(*nCol)*sizeof(double), ALIGNT);
  A = malloc((*nRow)*(*nCol)*sizeof(double));
  if(A == NULL)
  {
    printf("! ERREUR ! Allocation mémoire échouée !\n");
  }

  for(int i = 0; i < nz; i++)
  {
    int l, c;
    fscanf(file, "%d", &l);
    fscanf(file, "%d", &c);
    fscanf(file, "%lg\n", &(A[(l - 1)*(*nCol) + c - 1]));
  }

  *ptrMat = A;

  if(file != stdin)
    fclose(file);

  return;
}


int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  srand(0);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);

  FILE* fichier = NULL;

  if (argc < 4)
  {
    fprintf(stderr, "Usage: %s matrix_file n_eigen_values max_iter [iter_qr]\n", argv[0]);
    exit(1);
  }

  double* A_in;
  int dim, dim2;
  if(atoi(argv[1]) == 0)
  {
    fichier = fopen(argv[1], "r");
    if (fichier == NULL)
    {
      printf("! ERREUR ! Ouverture du fichier échouée !");
      exit(1);
    }
    load_mat(&A_in, &dim, &dim2, fichier);
  }

  //if(rank == 0) print_mat_format(A_in, 8, 8, 2, ROW_MAJOR, ", ", ",\n", "{", "}", "{\n", "\n}\n");
 

  int n_eigen_values = atoi(argv[2]);
  int subspace_dim = n_eigen_values * 2;
  int max_iter = atoi(argv[3]);
  int iter_qr = -1;
  if(argc == 5)
    iter_qr = atoi(argv[4]);
  double residual_max = 0.00000000000000000001;
  int n_rows_A = dim;
  int n_cols_A = dim;
  int n_local_rows_A;
  int n_local_cols_A;
  int n_local_rows_V;
  int n_local_cols_V;
  int n_local_rows_H;
  int n_local_cols_H;
  double* A = NULL;
  double* H = NULL;
  double* V = NULL;
  int* n_rows_V_per_process;
  int n_iter;
  int desired = n_eigen_values;
  double* residuals = malloc(max_iter * sizeof(double));
  double* eigen_values_r = malloc(subspace_dim * sizeof(double));
  double* eigen_values_i = malloc(subspace_dim * sizeof(double));
  double (*eigen_vectors)[subspace_dim*2] = malloc(2 * subspace_dim * subspace_dim * sizeof(double));
  memset(residuals, 0, max_iter * sizeof(double));
  memset(eigen_values_r, 0, subspace_dim * sizeof(double));
  memset(eigen_values_i, 0, subspace_dim * sizeof(double));
  memset(eigen_vectors, 0, subspace_dim * subspace_dim * 2 * sizeof(double));
  
  struct timeval begin_Arnoldi;
  struct timeval end_Arnoldi;
  struct timeval begin_IRAM;
  struct timeval end_IRAM;

  init_work(MPI_COMM_WORLD, n_rows_A, n_cols_A, subspace_dim, (double*) A_in, &n_local_rows_A, &n_local_cols_A, &n_local_rows_V, &n_local_cols_V, &n_rows_V_per_process, &n_local_rows_H, &n_local_cols_H, &A, &H, &V);

  //First Arnoldi Factorization
  if(rank == 0) gettimeofday(&begin_Arnoldi, NULL);
  Arnoldi_Factorization(A, n_local_rows_A, n_local_cols_A,
                        V, n_local_rows_V, n_local_cols_V, n_rows_V_per_process,
                        H, n_local_rows_H, n_local_cols_H,
                        subspace_dim, 0, MPI_COMM_WORLD);
  if(rank == 0) gettimeofday(&end_Arnoldi, NULL);

  //IRAM
  if(rank == 0) gettimeofday(&begin_IRAM, NULL);
  IRAM(A, n_local_rows_A, n_local_cols_A,
                            V, n_local_rows_V, n_local_cols_V, n_rows_V_per_process,
                            H, n_local_rows_H, n_local_cols_H,
                            subspace_dim, max_iter, desired, residuals, &n_iter, iter_qr, residual_max, eigen_values_r, eigen_values_i, (double*) eigen_vectors, MPI_COMM_WORLD);
  if(rank == 0) gettimeofday(&end_IRAM, NULL);

  if(rank == 0)
  {
    if(end_Arnoldi.tv_usec - begin_Arnoldi.tv_usec < 0)
    {
      fprintf(stderr, "Arnoldi time = %ld : %.6ld\n", end_Arnoldi.tv_sec - begin_Arnoldi.tv_sec - 1, end_Arnoldi.tv_usec - begin_Arnoldi.tv_usec + 1000000);
    }
    else
    {
      fprintf(stderr, "Arnoldi time = %ld : %.6ld\n", end_Arnoldi.tv_sec - begin_Arnoldi.tv_sec, end_Arnoldi.tv_usec - begin_Arnoldi.tv_usec);
    }

    if(end_IRAM.tv_usec - begin_IRAM.tv_usec < 0)
    {
      fprintf(stderr, "IRAM time = %ld : %.6ld\n", end_IRAM.tv_sec - begin_IRAM.tv_sec - 1, end_IRAM.tv_usec - begin_IRAM.tv_usec + 1000000);
    }
    else
    {
      fprintf(stderr, "IRAM time = %ld : %.6ld\n", end_IRAM.tv_sec - begin_IRAM.tv_sec, end_IRAM.tv_usec - begin_IRAM.tv_usec);
    }

    fprintf(stderr, "IRAM: %d restarts\n", n_iter);
    fprintf(stderr, "Eigen values:\n");
    for(int i = 0; i < n_eigen_values; ++i)
    {
        //fprintf(stderr, "%.10f + %.10fi\n", eigen_values_r[i], eigen_values_i[i]);
        fprintf(stderr, "%.10f\n", eigen_values_r[i]);
    }
  }

  free(residuals);
  free(eigen_values_r);
  free(eigen_values_i);
  free(eigen_vectors);

  MPI_Finalize();
}
