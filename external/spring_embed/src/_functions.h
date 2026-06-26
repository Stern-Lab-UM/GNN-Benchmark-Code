#include <stdio.h>
#include <cstring>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>
#define LINE_SIZE 512


//vertices
double **v;
//.................
int **v_edges;
int **v_cells;
//.................
double **v_F;
int *v_T1edge;
double *v_clock;
int **v_resolveCells;


//edges
int **e;
//.................
int **e_cells;
int **e_Vcells;
//.................
double *e_g;
double *e_Fg;
double *e_l0;
double *e_linitial;
double *e_length;
double *e_dl;
double *e_inLen;
double *e_outLen;
int *e_wasFlipped;


//cells
int **c;
//.................
int **c_vertices;
int **c_Vedges;
//.................
double *c_area;
double *c_A0;



//MISC
int Nv, Ne, Nc;
double *perioXYZ;
int array_max;
int **ids, **exist;
double wA, wP, wL;

#include "_allocate.h"
#include "_arrays.h"
#include "_free_id.h"
#include "_torus.h"
#include "_dissolve.h"
#include "_create.h"
#include "_initial_structure.h"
#include "_rndom.h"
#include "_T1_edge_to_vertex.h"
#include "_T1_vertex_to_edge.h"
#include "_force_length.h"
#include "_force_area.h"
#include "_output.h"
#include "_equation_of_motion.h"
#include "_T1_transition.h"
#include "_cell_extrusion.h"
