//****************************************************************************
//****************************************************************************
//****************************T1 TRANSITION***********************************
//****************************************************************************
//****************************************************************************
int vert_edg_cells_vte(int i, int *vertices, int *edges, int *cells)
{
    //VERTICES
    appendToList(vertices,i);

    //CELLS
    for(int j=1; j<=v_resolveCells[i][0]; j++) appendToList(cells,v_resolveCells[i][j]);
    for(int j = 1; j<=v_cells[i][0]; j++) appendToList(cells,v_cells[i][j]);

    //EDGES
    int countEdges=0;
    int c1=cells[1];
    for(int j = 1; j<=c[c1][0]; j++){
        int eID=abs(c[c1][j]);
        int v1=e[eID][1];
        int v2=e[eID][2];
        if(v1==i || v2==i){
            countEdges++;
            appendToList(edges,eID);
            /*CELLS*/for(int k=1; k<=e_Vcells[eID][0]; k++) appendToList(cells,e_Vcells[eID][k]);
        }
    }
    for(int j = 1; j<=v_edges[i][0]; j++) appendToList(edges,v_edges[i][j]);

    return countEdges;
}
//****************************************************************************
void direction_vector(int i, int cID, double *cdir)
{
    double *dxdydz = new double[3];
    double *centXYZ= new double[3]; centXYZ[1]=0; centXYZ[2]=0;

    //CENTER OF CELL cID
    int ref_vID=c_vertices[cID][1]; for(int k=1; k<=2; k++) centXYZ[k]+=v[ref_vID][k];
    for(int j=2; j<=c_vertices[cID][0]; j++){
        int vID=c_vertices[cID][j];
        torus_dx_dy_dz(dxdydz,vID,ref_vID);
        for(int k=1; k<=2; k++) centXYZ[k] += v[vID][k] + dxdydz[k];
    }
    for(int k=1; k<=2; k++) centXYZ[k]/=(1.*c_vertices[cID][0]);

    //DIRECTION OF RESOLUTION
    for(int k=1; k<=2; k++){
        if(fabs(centXYZ[k]-v[i][k])>perioXYZ[k]/2.){
            if(centXYZ[k]>v[i][k]) cdir[k]=centXYZ[k]-(v[i][k]+perioXYZ[k]);
            if(centXYZ[k]<v[i][k]) cdir[k]=centXYZ[k]-(v[i][k]-perioXYZ[k]);
        }
        else cdir[k]=centXYZ[k]-v[i][k];
    }
    double norm=sqrt(cdir[1]*cdir[1]+cdir[2]*cdir[2]);
    for(int k=1; k<=2; k++) cdir[k]/=norm;

    delete [] dxdydz;
    delete [] centXYZ;
}
//****************************************************************************
int create_new_vertex(int i, int *cells, double fin_len){

    double *cdir; cdir=new double[3];

    //vertex v2
    direction_vector(i,cells[1],cdir);
    int v2=make_vertex(v[i][1]+0.5*fin_len*cdir[1],
                       v[i][2]+0.5*fin_len*cdir[2]);

    //vertex i
    for(int k=1; k<=2; k++) v[i][k]-=0.5*fin_len*cdir[k];
    torus_vertex(i);

    delete [] cdir;
    return v2;
}
//****************************************************************************
void restitch_edges_vte(int *vertices, int *edges, int countEdges)
{
    for(int i=1; i<=countEdges; i++) remake_edge(edges[i],vertices[1],vertices[2]);
}
//****************************************************************************
void insertEdgeIntoCell(int i, int j, int el)
{
    c[i][0]++;
    if(j==c[i][0]) c[i][j]=el;
    else {
        for(int k=c[i][0]; k>j; k--) c[i][k]=c[i][k-1];
        c[i][j]=el;
    }
}
//****************************************************************************
void correct_cells(int cID, int newe)
{
    int v1=e[newe][1]; int v2=e[newe][2];
    for(int k=1; k<=c[cID][0]; k++){
        //vprev
        int oeID=c[cID][k];
        int vprev=0;
        if(oeID>0) vprev=e[abs(oeID)][2];
        else vprev=e[abs(oeID)][1];
        //vnext
        if(k==c[cID][0]) oeID=c[cID][1];
        else oeID=c[cID][k+1];
        int vnext=0;
        if(oeID>0) vnext=e[abs(oeID)][1];
        else vnext=e[abs(oeID)][2];
        if(vprev!=vnext){
            if      (vprev==v1 && vnext==v2)   insertEdgeIntoCell(cID,k+1,newe);
            else if (vprev==v2 && vnext==v1)   insertEdgeIntoCell(cID,k+1,-newe);
            break;
        }
    }
}
//****************************************************************************
int T1_VERTEX_TO_EDGE(int i, double fin_len)
{

    //******************************************************************************
    int *vertices; vertices = new int[200]; for(int j=0; j<=199; j++) vertices[j]=0;
    int *edges; edges = new int[200]; for(int j=0; j<=199; j++) edges[j]=0;
    int *cells; cells = new int[200]; for(int j=0; j<=199; j++) cells[j]=0;
    int countEdges=vert_edg_cells_vte(i,vertices,edges,cells);
//    printf("vertices\n"); printLIST(vertices);
//    printf("edges\n"); printLIST(edges);
//    printf("cells\n"); printLIST(cells);
//    printf("resolution cells\n"); printARRAY(v_resolveCells,i);
    //******************************************************************************


    //MAKE VERTEX
    int v2=create_new_vertex(i,cells,fin_len);
    appendToList(vertices,v2);


    //DISSOLVE OBJECTS
    for(int j=1; j<=cells[0]; j++) dissolve_cell(cells[j]);


    //RESTITCH EDGES
    restitch_edges_vte(vertices,edges,countEdges);


    //MAKE EDGE
    int newe=make_edge(vertices[1],vertices[2],v_T1edge[i]); v_T1edge[i]=0;
    reset_edge_attributes(newe);


    //INSERT EDGE INTO CELLS (nujno za restitch in make)
    for(int j=1; j<=cells[0]; j++) correct_cells(cells[j],newe);


    //MAKE OBJECTS
    for(int j=1; j<=cells[0]; j++) make_cell(cells[j]);


    //v_resolveCells
    v_resolveCells[i][0]=0;


    //******************************************************************************
    delete [] vertices;
    delete [] edges;
    delete [] cells;
    //******************************************************************************

    return newe;
}
//****************************************************************************
