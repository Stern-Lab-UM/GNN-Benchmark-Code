//****************************************************************************
//****************************************************************************
//****************************T1 TRANSITION***********************************
//****************************************************************************
//****************************************************************************
/*
 * vert_edg_cells_etv: Compute or update cell-level topology/geometry information.
 * Parameters: int i, int *vertices, int *edges, int *cells.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int vert_edg_cells_etv(int i, int *vertices, int *edges, int *cells)
{
    //VERTICES
    int v1=e[i][1]; appendToList(vertices,v1);
    int v2=e[i][2]; appendToList(vertices,v2);

    //EDGES
    appendToList(edges,i);
    for(int j = 1; j<=v_edges[v1][0]; j++) appendToList(edges,v_edges[v1][j]);
    for(int j = 1; j<=v_edges[v2][0]; j++) appendToList(edges,v_edges[v2][j]);

    //CELLS
    int nrCells=0;
    for(int j = 1; j<=e_cells[i][0]; j++) { appendToList(cells,e_cells[i][j]); nrCells++;}
    for(int j = 1; j<=v_cells[v1][0]; j++) appendToList(cells,v_cells[v1][j]);
    for(int j = 1; j<=v_cells[v2][0]; j++) appendToList(cells,v_cells[v2][j]);
    for(int j = 1; j<=v_edges[v2][0]; j++){
        int eID=v_edges[v2][j];
        for(int k=1; k<=e_Vcells[eID][0]; k++) appendToList(cells,e_Vcells[eID][k]);
    }

    //RESOLUTION CELLS
    for(int j=1; j<=e_cells[i][0]; j++) appendToArray(v_resolveCells,vertices[1],e_cells[i][j]);

    return nrCells;
}
//****************************************************************************
/*
 * restitch_edges_etv: Compute or update edge-level topology/geometry information.
 * Parameters: int *vertices, int *edges, int valv2.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void restitch_edges_etv(int *vertices, int *edges, int valv2)
{
    for(int j=0; j<=valv2-2; j++) remake_edge(edges[edges[0]-j],vertices[2],vertices[1]);
}
//****************************************************************************
/*
 * merge_vertices: Implement the merge vertices operation for the C vertex-model code.
 * Parameters: int *vertices.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void merge_vertices(int *vertices)
{
    int v1=vertices[1];
    int v2=vertices[2];
    for(int k=1; k<=2; k++){
        double XYZ;
        if(fabs(v[v2][k]-v[v1][k])>perioXYZ[k]/2.) XYZ=0.5*(v[v1][k]+v[v2][k]+perioXYZ[k]);
        else XYZ=0.5*(v[v1][k]+v[v2][k]);
        v[v1][k]=XYZ;
    }
    torus_vertex(v1);
}
//****************************************************************************
/*
 * T1_EDGE_TO_VERTEX: Apply or convert a T1 transition in the vertex-model topology.
 * Parameters: int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int T1_EDGE_TO_VERTEX(int i)
{

    //******************************************************************************
    int *vertices; vertices = new int[200]; for(int j=0; j<=199; j++) vertices[j]=0;
    int *edges; edges = new int[200]; for(int j=0; j<=199; j++) edges[j]=0;
    int *cells; cells = new int[200]; for(int j=0; j<=199; j++) cells[j]=0;
    int nrCells=vert_edg_cells_etv(i,vertices,edges,cells);
    int valv2=v_edges[vertices[2]][0];
//    printf("vertices\n"); printLIST(vertices);
//    printf("edges\n"); printLIST(edges);
//    printf("cells\n"); printLIST(cells);
//    printf("resolution cells\n"); printARRAY(v_resolveCells,vertices[1]);
    //******************************************************************************


    //DISSOLVE OBJECTS
    for(int j=1; j<=cells[0]; j++) dissolve_cell(cells[j]);


    //REMOVE EDGE i FROM LIST OF CELLS
    for(int j=1; j<=nrCells; j++) removeELEMENT(c,cells[j],i);


    //DISSOLVE EDGE
    dissolve_edge(i,1);


    //MERGE VERTEX POSITIONS
    merge_vertices(vertices);


    //RESTITCH EDGES
    restitch_edges_etv(vertices,edges,valv2);


    //DISSOLVE VERTEX v2
    dissolve_vertex(vertices[2]);


    //MAKE OBJECTS
    for(int j=1; j<=cells[0]; j++) make_cell(cells[j]);


    //******************************************************************************
    int v1=vertices[1]; v_T1edge[v1]=i;
    delete [] vertices;
    delete [] edges;
    delete [] cells;
    //******************************************************************************


    return v1;
}
//****************************************************************************
