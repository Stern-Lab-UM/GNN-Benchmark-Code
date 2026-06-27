//****************INPUT PARAMETERS************************
double kA=0;    // embedding is length-only (paper p.16: P=Sum(l-L)^2); area incompressibility is forward-sim only
double kS=1.;

//OTHER PARAMETERS
double P0=3.8;
double kM=1.;
double Sig=0;
double Time=0;
int nSteps=2000;
double h0=0.001;
double lth=0.01;
double h;

//*******************DECLARATIONS*************************
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#define DCG_MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define DCG_MKDIR(path) mkdir(path, 0777)
#endif

#include "_functions.h"
/*
 * ensure_output_dir: Write vertex-model state or graph data to output files.
 * Parameters: none.
 * Returns: see the C signature; most routines update global vertex-model state.
 */

static int ensure_output_dir(void)
{
    int rc = DCG_MKDIR("output");
    if (rc == 0 || errno == EEXIST) return 1;

    perror("Could not create output directory");
    return 0;
}

//********************MAIN********************************
// Usage: spring_embed <initial.vt2d> <prediction_file> <target_sim_id>
// Reads col6 (predicted_length) -> e_l0 and col4 (was_flipped); to embed GT or the
// baseline, feed a prediction file whose col6 has been replaced by col5 / col3.
// All outputs (X_/Y_/out_) go to ./output relative to the working directory.
/*
 * main: Run the command-line entry point for this C executable.
 * Parameters: int argc, char *argv[].
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int main(int argc, char *argv[]){


    //INITIALIZE
    array_max=100000;
    allocate();
    if (!ensure_output_dir()) return 1;
    const char *initial_vt2d   = (argc>1)? argv[1] : "./initial/tomer_data/initial/final_256_17_0.vt2d";
    const char *prediction_file = (argc>2)? argv[2] : "./initial/tomer_data/gnn_GraphSAGE_weighted_True (predictions).txt";
    const char *target_sim_id  = (argc>3)? argv[3] : "graph_256_17_0.txt";
    set_initial_fromFile(initial_vt2d,
                         prediction_file,
                         target_sim_id
                         );
    out_tissue(1);
    //*************************


    //FLIP
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1 && e_wasFlipped[i]>0.5){
        printf("%d  %d  %d\n", i, e_cells[i][1], e_cells[i][2]);
        int vID=T1_EDGE_TO_VERTEX(i);
        int newe=T1_VERTEX_TO_EDGE(vID,0.001);
        e_l0[newe]=e_l0[0];
        e_outLen[newe]=e_outLen[0];
        printf("%d  %d  %d\n", newe, e_cells[newe][1], e_cells[newe][2]);

    }
    out_tissue(2);
    //*************************


    //DYNAMICS
    Time=0; for(int iter=0; iter<nSteps; iter++) eqOfMotion();
    out_tissue(3);
    //*************************


    //OUPTUT
    char path[256]; snprintf(path, sizeof(path), "./output/out_%s", target_sim_id);
    FILE *filee = fopen(path, "wt");
    if (filee == NULL) {
        fprintf(stderr, "Error opening file: %s\n", path);
        return 1;
    }
    for(int i=1; i<=Ne; i++) if(exist[2][i]==1) fprintf(filee, "%d  %d  %g  %g  %g\n", e_cells[i][1], e_cells[i][2], e_outLen[i], e_l0[i], edge_length(i));
    fclose(filee);
    //*************************


    //DEALLOCATE
    deallocate();
    //*************************


    return 0;
}
//********************************************************
