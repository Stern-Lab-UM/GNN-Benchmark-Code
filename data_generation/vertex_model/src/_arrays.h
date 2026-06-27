//****************************************************************************
//****************************************************************************
//******************************CREATE****************************************
//****************************************************************************
//****************************************************************************
/*
 * appendToArray: Implement the append to array operation for the C vertex-model code.
 * Parameters: int **aRray, int i, int el.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int appendToArray(int **aRray, int i, int el){

    //CRITICAL ERRORS
    if(i==0) { printf("%g| ERROR: array id=0!\n", Time);   exit(0); }
    if(el==0){ printf("%g| ERROR: element id=0!\n", Time); exit(0); }

    //CHECK ELEMENTS OF aRray
    for(int k=1; k<=aRray[i][0]; k++){
        if( el==aRray[i][k] ) return 0;
        if( el==-aRray[i][k] ) { printf("%g| ERROR: this element already present but with different sign!\n", Time); exit(0); }
    }

    //ADD ELEMENT TO aRray
    aRray[i][0]++; aRray[i][aRray[i][0]]=el; return 1;
}
//****************************************************************************
/*
 * appendToArrayDouble: Implement the append to array double operation for the C vertex-model code.
 * Parameters: double **aRray, int i, double el.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int appendToArrayDouble(double **aRray, int i, double el){
    aRray[i][0]++; aRray[i][(int)aRray[i][0]]=el;
    return 1;
}
//****************************************************************************
/*
 * appendToList: Implement the append to list operation for the C vertex-model code.
 * Parameters: int *lIst, int el.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int appendToList(int *lIst, int el){

    //CRITICAL ERRORS
    if(el==0){ printf("%g| ERROR: element id=0!\n", Time); exit(0); }

    //CHECK ELEMENTS OF lIst
    for(int k=1; k<=lIst[0]; k++){
        if( el==lIst[k] ) return 0;
        if( el==-lIst[k] ) { printf("%g| ERROR: this element already present but with different sign!\n", Time); exit(0); }
    }

    //ADD ELEMENT TO aRray
    lIst[0]++; lIst[lIst[0]]=el; return 1;
}
//****************************************************************************
/*
 * isItin: Implement the is itin operation for the C vertex-model code.
 * Parameters: int **aRray, int i, int el.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int isItin(int **aRray, int i, int el){
    for(int k=1; k<=aRray[i][0]; k++){
        if( aRray[i][k]==el )   return 1;
        if( aRray[i][k]==-el )  return -1;
    }
    return 0;
}
//****************************************************************************
/*
 * isItinList: Implement the is itin list operation for the C vertex-model code.
 * Parameters: int *lIst, int el.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int isItinList(int *lIst, int el){
    for(int k=1; k<=lIst[0]; k++){
        if( lIst[k]==el )   return 1;
        if( lIst[k]==-el )  return -1;
    }
    return 0;
}
//****************************************************************************
/*
 * removeELEMENT: Implement the remove element operation for the C vertex-model code.
 * Parameters: int **aRray, int i, int el.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int removeELEMENT(int **aRray, int i, int el){

    //CRITICAL ERRORS
    if(i==0) { printf("%g| ERROR: array id=0!\n", Time);   exit(0); }
    if(el==0){ printf("%g| ERROR: element id=0!\n", Time); exit(0); }

    //CHECK ELEMENTS OF aRray AND REMOVE el
    for(int k=1; k<=aRray[i][0]; k++) if( aRray[i][k]==el || aRray[i][k]==-el ) {
        for(int l=k; l<aRray[i][0]; l++) aRray[i][l]=aRray[i][l+1];
        aRray[i][0]--;
        return 1;
    }

    printf("%g| ERROR: element not found in the array!\n", Time);   exit(0);
}
//****************************************************************************
/*
 * printARRAY: Write vertex-model state or graph data to output files.
 * Parameters: int **aRray, int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void printARRAY(int **aRray, int i){
    printf("%d  ||  ", aRray[i][0]);
    for(int j=1; j<=aRray[i][0]; j++) printf("%d  ", aRray[i][j]);
    printf("\n");
}
//****************************************************************************
/*
 * printLIST: Write vertex-model state or graph data to output files.
 * Parameters: int *lIst.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void printLIST(int *lIst){
    printf("%d  ||  ", lIst[0]);
    for(int j=1; j<=lIst[0]; j++) printf("%d  ", lIst[j]);
    printf("\n");
}
//****************************************************************************
/*
 * status_vertices: Implement the status vertices operation for the C vertex-model code.
 * Parameters: none.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void status_vertices(){
    printf("VERTICES\n");
    printf("Nv=%d\n", Nv);
    printf("Free ids:  "); for(int j=1; j<=ids[1][0]; j++) printf("%d  ", ids[1][j]); printf("\n");
    int nrExist=0; for(int j=1; j<=array_max; j++) if(exist[1][j]==1) nrExist++;
    printf("Dissolved: %d\n", Nv-nrExist);
}
//****************************************************************************
/*
 * status_edges: Compute or update edge-level topology/geometry information.
 * Parameters: none.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void status_edges(){
    printf("\nEDGES\n");
    printf("Ne=%d\n", Ne);
    printf("Free ids:  "); for(int j=1; j<=ids[2][0]; j++) printf("%d  ", ids[2][j]); printf("\n");
    int nrExist=0; for(int j=1; j<=array_max; j++) if(exist[2][j]==1) nrExist++;
    printf("Dissolved: %d\n", Ne-nrExist);
}
//****************************************************************************
/*
 * status_cells: Compute or update cell-level topology/geometry information.
 * Parameters: none.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void status_cells(){
    printf("\nCELLS\n");
    printf("Nc=%d\n", Nc);
    printf("Free ids:  "); for(int j=1; j<=ids[3][0]; j++) printf("%d  ", ids[3][j]); printf("\n");
    int nrExist=0; for(int j=1; j<=array_max; j++) if(exist[3][j]==1) nrExist++;
    printf("Dissolved: %d\n", Nc-nrExist);
}
//****************************************************************************
/*
 * print_status: Write vertex-model state or graph data to output files.
 * Parameters: none.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void print_status(){
    printf("STATUS:\n\n");
    status_vertices();
    status_edges();
    status_cells();
}
//****************************************************************************
//    printf("%d  ||  ", p[pID][0]);
//    for(int l=1; l<=p[pID][0]; l++){
//        if(p[pID][l]>0) printf("%d--(%d)-->%d  ", e[abs(p[pID][l])][1], p[pID][l], e[abs(p[pID][l])][2]);
//        else printf("%d<--(%d)--%d  ", e[abs(p[pID][l])][2], p[pID][l], e[abs(p[pID][l])][1]);
//    }
//    printf("\n");
