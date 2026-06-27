//****************************************************************************
//****************************************************************************
//****************************T1 TRANSITION***********************************
//****************************************************************************
//****************************************************************************
void division_plane(int i, double *Px, double *Py, double *dirX, double *dirY){
    //CALCULATE CELL CENTER
    double *dxdydz = new double[3];
    int ref_vID=c_vertices[i][1];
    double vrefx = v[ref_vID][1];
    double vrefy = v[ref_vID][2];
    double sumX=0;
    double sumY=0;
    for(int j=1; j<=c_vertices[i][0]; j++ ){
        int vID=c_vertices[i][j];
        torus_dx_dy_dz(dxdydz,vID,ref_vID);
        double vx=v[vID][1]+dxdydz[1];
        double vy=v[vID][2]+dxdydz[2];
        sumX+=vx/(1.*c_vertices[i][0]);
        sumY+=vy/(1.*c_vertices[i][0]);
    }
    *Px=sumX;
    *Py=sumY;
    delete []dxdydz;

    //PICK RANDOM ANGLE
    double phi=M_PI*rnd();
    *dirX=cos(phi);
    *dirY=sin(phi);
}
//****************************************************************************
int intersection_vertices(int i, int **newVertices){

    //DIVISION PLANE
    double dirX, dirY, Px, Py;
    division_plane(i,&Px,&Py,&dirX,&dirY);
    printf("%g  %g\n", Px, Py);
    printf("%g  %g\n", dirX, dirY);

    //FIND PLANE-EDGE INTERSECTIONS
    double *dxdydz = new double[3];
    int nrIntrsct=0;
    int ref_vID=c_vertices[i][1];
    for(int j=1; j<=c[i][0]; j++){
        int eID=abs(c[i][j]);
        double elen=edge_length(eID);

        int v1=e[eID][1];
        int v2=e[eID][2];

        //POSITIONS IN THE ORIGINAL REFERENCE FRAME
        double x1, y1;
        double x2, y2;
        //v1
        int vID=v1;
        torus_dx_dy_dz(dxdydz,vID,ref_vID);
        x1=v[vID][1]+dxdydz[1]; y1=v[vID][2]+dxdydz[2];
        //v2
        vID=v2;
        torus_dx_dy_dz(dxdydz,vID,ref_vID);
        x2=v[vID][1]+dxdydz[1]; y2=v[vID][2]+dxdydz[2];

        //POSITIONS RELATIVE TO THE REFERENCE POINT
        double X1=x1-Px, Y1=y1-Py;
        double X2=x2-Px, Y2=y2-Py;

        //PROJECTION ON NORMAL VECTOR
        double scalar1=dirY*X1-dirX*Y1;
        double scalar2=dirY*X2-dirX*Y2;

        //PLANE-EDGE INTERSECTION
        if(scalar1*scalar2<0){
            printf("%d\n", j);
            double t=((Px-x1)*(-dirY)+(Py-y1)*dirX)/((x2-x1)*(-dirY)+(y2-y1)*dirX);
            double xNEW=x1+t*(x2-x1);
            double yNEW=y1+t*(y2-y1);
            printf("%g  %g\n", xNEW, yNEW);
            //create vertex
            nrIntrsct++;
            int vNEW=make_vertex(xNEW,yNEW);
            newVertices[nrIntrsct][1]=eID;
            newVertices[nrIntrsct][2]=vNEW;
        }
    }
    delete []dxdydz;

    //RETURN
    return nrIntrsct;
}
//****************************************************************************
int edg_cells_div(int i, int nrIntrsct, int **newVertices, int *edges, int *cells)
{
    //EDGES
    for(int j=1; j<=nrIntrsct; j++) appendToList(edges,newVertices[j][1]);

    //CELLS
    appendToList(cells,i);
    for(int j=1; j<=edges[0]; j++){
        int eID=edges[j];
        for(int k=1; k<=e_cells[eID][0]; k++) appendToList(cells,e_cells[eID][k]);
    }
    for(int j=1; j<=edges[0]; j++){
        int eID=edges[j];
        for(int k=1; k<=e_Vcells[eID][0]; k++) appendToList(cells,e_Vcells[eID][k]);
    }

    printf("edges\n"); printLIST(edges);
    printf("cells\n"); printLIST(cells);

    return 0;
}
//****************************************************************************
int restitch_edge_div(int j, int **newVertices, int *edges){
    int eID=newVertices[j][1];
    int v1=e[eID][1];
    int v2=e[eID][2];
    int vNEW=newVertices[j][2];

    //eID
    remake_edge(eID,v2,vNEW);

    //eNEW
    int eNEW=make_edge(vNEW,v2,0);
    newVertices[j][3]=eNEW;
    appendToList(edges,eNEW);

    //RETURN
    return eNEW;
}
//****************************************************************************
int check_corrected_cells_div(int cID){
    int vinit, vprev, v1, v2, eID;
    for(int j=1; j<=c[cID][0]; j++){
        eID=c[cID][j];
        if(eID>0){
            v1=e[abs(eID)][1];
            v2=e[abs(eID)][2];
        }
        else{
            v1=e[abs(eID)][2];
            v2=e[abs(eID)][1];
        }
        if(j==1){
            vinit=v1;
            vprev=v2;
        }
        else{
            if(v1==vprev){
                vprev=v2;
            }
            else { printf("ERROR in correct_polygon_CD\n"); exit(0);}
        }
    }
    if(vprev!=vinit) { printf("ERROR in correct_polygon_CD\n"); exit(0);}
    return 1;
}
//****************************************************************************
void correct_cells_div(int *cells, int nrIntrsct, int **newVertices){
    for(int j=1; j<=cells[0]; j++){
        int cID=cells[j];
        for(int l=1; l<=nrIntrsct; l++){
            int edge=newVertices[l][1];
            for(int k=1; k<=c[cID][0]; k++){
                int eID=c[cID][k];
                if(eID==edge){
                    insertEdgeIntoCell(cID,k+1,newVertices[l][3]);
                    break;
                }
                if(-eID==edge){
                    insertEdgeIntoCell(cID,k,-newVertices[l][3]);
                    break;
                }
            }
        }
        //CHECK
        check_corrected_cells_div(cID);
    }
}
//****************************************************************************
int correct_intersected_polygon_div(int i, int newe, int *cells){
    //ALLOCATE
    int *p1; p1 = new int[26]; p1[0]=0;
    int *p2; p2 = new int[26]; p2[0]=0;

    //MAIN BODY
    int V1=e[newe][1], V2=e[newe][2];
    int vTarget=0;
    int neweSIGN=0;
    for(int k=1; k<=c[i][0]; k++){
        int eID=c[i][k];
        int v1=e[abs(eID)][1];
        int v2=e[abs(eID)][2];
        if(eID<0){
            v1=e[abs(eID)][2];
            v2=e[abs(eID)][1];
        }
        if(vTarget==0) appendToList(p1,eID);
        else if(v1==vTarget){
            appendToList(p1,eID);
            vTarget=0;
        }
        else appendToList(p2,eID);
        if(vTarget==0){
            if(V1==v2){
                appendToList(p1,newe);
                neweSIGN=1;
                vTarget=V2;
            }
            if(V2==v2){
                appendToList(p1,-newe);
                neweSIGN=-1;
                vTarget=V1;
            }
        }

    }
    appendToList(p2,-neweSIGN*newe);

    //p1
    for(int j=0; j<=p1[0]; j++) c[i][j]=p1[j];
    check_corrected_cells_div(i);
    //p2
    int newPoly=cell_sides(p2[0],p2,0); appendToList(cells,newPoly);
    check_corrected_cells_div(newPoly);

    //DEALLOCATE
    delete [] p1;
    delete [] p2;

    //RETURN
    return newPoly;
}
//****************************************************************************
int CELL_DIVISION(int i)
{

    printARRAY(c,i);

    //******************************************************************************
    int *vertices; vertices = new int[200]; for(int j=0; j<=199; j++) vertices[j]=0;
    int *edges; edges = new int[200]; for(int j=0; j<=199; j++) edges[j]=0;
    int *cells; cells = new int[200]; for(int j=0; j<=199; j++) cells[j]=0;
    //newVertices
    int **newVertices;  newVertices = new int*[array_max+1];
    for(int k=0; k<=array_max; k++) newVertices[k] = new int[4];
    for(int k=0; k<=array_max; k++) for(int j=0; j<=3; j++) newVertices[k][j]=0;
    //******************************************************************************

    //CREATE INTERSECTION VERTICES
    int nrIntrsct=intersection_vertices(i,newVertices);

    //IDENTIFY EDGES AND CELLS
    edg_cells_div(i,nrIntrsct,newVertices,edges,cells);

    //DISSOLVE CELLS
    for(int j=1; j<=cells[0]; j++) dissolve_cell(cells[j]);

    //RESTITCH EDGES
    for(int j=1; j<=nrIntrsct; j++) restitch_edge_div(j,newVertices,edges);

    //CORRECT CELL
    correct_cells_div(cells,nrIntrsct,newVertices);

    //CREATE INTERSECTING EDGE
    int newe=make_edge(newVertices[1][2],newVertices[2][2],0);

    //CORRECT INTERSECTED POLYGON
    int newc=correct_intersected_polygon_div(i,newe,cells);

    //MAKE OBJECTS
    for(int j=1; j<=cells[0]; j++) make_cell(cells[j]);

    printARRAY(c,i);
    printARRAY(c,newc);

    //******************************************************************************
    delete [] vertices;
    delete [] edges;
    delete [] cells;
    //newVertices
    for(int k=0; k<=array_max; k++) delete [] newVertices[k]; delete [] newVertices;
    //******************************************************************************


    return 0;
}
//****************************************************************************
