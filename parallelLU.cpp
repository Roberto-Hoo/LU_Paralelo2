//parallelLU.cpp
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <ctime>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <iomanip>

const int N = 6;
const int N1 = 8;
int nlin;
int ncol;
using namespace std;

char caracter;
int n = 5;
int i, j;
gsl_matrix *A = gsl_matrix_alloc(n, n);
gsl_matrix *U = gsl_matrix_alloc(n, n);
gsl_matrix *L = gsl_matrix_alloc(n, n);

double **M = nullptr;
double **MU = nullptr;
double **ML = nullptr;
double **B = nullptr;

unsigned seed;

void identidade(int n, double **&A);

int numeroAleatorio(int menor, int maior);

void geraMatriz(int min, int max, int nlin, int ncol, double **&M);

void copia(int nlin, int ncol, double **&A, double **&U);

void geraLU(int n, double **&A, double **&L, double **&U);

//void printArray(char ,double a[][N]) {
void imprimeMatriz(char *nome, int nlin, int ncol, double **&A);

void CriaMatriz(int nlin, int ncol, double **&M);

void DeleteData();

void multiplica(int n, double **&L, double **&U, double **&B);

int main(int argc, char *argv[]) {


    // gerador randÃ´mico
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, time(NULL));

    // inicializaÃ§Ã£o
    printf("Inicializando ... \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            int sig = 1;
            if (gsl_rng_uniform(rng) >= 0.5)
                sig = -1;
            gsl_matrix_set(A, i, j, sig * gsl_rng_uniform(rng));
            gsl_matrix_set(A, j, i, gsl_matrix_get(A, i, j));
        }
        int sig = 1;
        if (gsl_rng_uniform(rng) >= 0.5)
            sig = -1;
        gsl_matrix_set(A, i, i, sig * gsl_rng_uniform_pos(rng));
    }
    // Print the values of A and b using GSL print functions
    cout << "A = \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //printf("A(%d,%d) = %g\n", i, j, gsl_matrix_get(A, i, j));
            printf(" %10.6f  ", gsl_matrix_get(A, i, j));
        }
        printf("\n");
    }
    printf("feito.\n");

    // U = A
    gsl_matrix_memcpy(U, A);
    // L = I
    gsl_matrix_set_identity(L);

    omp_set_num_threads(n);

    for (int k = 0; k < n - 1; k++) {
#pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            gsl_matrix_set(L, i, k, gsl_matrix_get(U, i, k) / gsl_matrix_get(U, k, k));
            for (int j = k; j < n; j++) {
                gsl_matrix_set(U, i, j, gsl_matrix_get(U, i, j)
                                        - gsl_matrix_get(L, i, k) * gsl_matrix_get(U, k, j));
            }
        }
    }

    // Print the values of L using GSL print functions
    cout << "L = \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //printf("A(%d,%d) = %g\n", i, j, gsl_matrix_get(A, i, j));
            printf(" %10.6f  ", gsl_matrix_get(L, i, j));
        }
        printf("\n");
    }

    // Print the values of U using GSL print functions
    cout << "U = \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //printf("A(%d,%d) = %g\n", i, j, gsl_matrix_get(A, i, j));
            printf(" %10.6f  ", gsl_matrix_get(U, i, j));
        }
        printf("\n");
    }


    printf("feito.\n");


    gsl_matrix_free(A);
    gsl_matrix_free(U);
    gsl_matrix_free(L);
    gsl_rng_free(rng);


    //srand((unsigned)time(0)); //para gerar números aleatórios reais.
    srand(1); // seed=2021, gera sempre os mesmos números
    nlin = 24;
    ncol = 24;
    CriaMatriz(nlin, ncol, M);
    CriaMatriz(nlin, ncol, ML);
    CriaMatriz(nlin, ncol, MU);
    geraMatriz(-100, 100, nlin, ncol, M);
    geraLU(nlin, M, ML, MU);

    imprimeMatriz("Matriz L =", nlin, ncol, ML);
    imprimeMatriz("Matriz U =", nlin, ncol, MU);

    multiplica(nlin,ML,MU,B);
    imprimeMatriz("Matriz B =", nlin, ncol, B);
    imprimeMatriz("Matriz original A =", nlin, ncol, M);

    //cout << "\n\nTecle uma tecla e apos Enter para finalizar...\n";
    //cin >> caracter;
    DeleteData();
    return 0;
}

void multiplica(int n, double **&L, double **&U, double **&B) {
    if (B == nullptr)
        CriaMatriz(n, n, B);
    double s;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            s = 0;
            for (int k = 0; k < n; k++) {
               s=s+L[i][k]*U[k][j];
            }
            B[i][j]=s;
        }
}


void DeleteData() {

    delete[] M;
    delete[] ML;
    delete[] MU;
    delete[] B;

}

void identidade(int n, double **&A) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j)
                A[i][i] = 1;
            else
                A[i][j] = 0;
        }
}

int numeroAleatorio(int menor, int maior) {
    return rand() % (maior - menor + 1) + menor;
}

void geraMatriz(int min, int max, int nlin, int ncol, double **&M) {
    for (i = 0; i < nlin; i++)
        for (j = 0; j < ncol; j++)
            M[i][j] = numeroAleatorio(min, max);
}

void copia(int nlin, int ncol, double **&A, double **&U) {
    for (int i = 0; i < nlin; i++)
        for (int j = 0; j < ncol; j++)
            U[i][j] = A[i][j];
}

void geraLU(int n, double **&A, double **&L, double **&U) {
    identidade(n, L);
    copia(n, n, A, U);
    for (int j = 0; j < n - 1; j++)
        for (int i = j + 1; i < n; i++) {
            L[i][j] = U[i][j] / U[j][j];
            for (int k = j; k < n; k++)
                U[i][k] = U[i][k] - L[i][j] * U[j][k];
        }
}

//void printArray(char ,double a[][N]) {
void imprimeMatriz(char *nome, int nlin, int ncol, double **&A) {
    printf("\n  %s \n", nome);
    // loop through array's rows
    for (int i = 0; i < nlin; ++i) {
        // loop through columns of current row
        for (int j = 0; j < ncol; ++j)
            printf(" %3.0f", A[i][j]);
        cout << endl; // start new line of output
    } // end outer for
} // end function printArray


void CriaMatriz(int nlin, int ncol, double **&M) {
    M = new double *[nlin];
    for (int i = 0; i < nlin; i++)
        M[i] = new double[ncol];
}
