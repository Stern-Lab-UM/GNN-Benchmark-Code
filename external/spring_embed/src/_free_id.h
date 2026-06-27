//****************************************************************************
//****************************************************************************
//*****************************FREE ID****************************************
//****************************************************************************
//****************************************************************************
/*
 * getID: Implement the get id operation for the C vertex-model code.
 * Parameters: int vec.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
int getID(int vec){
    if(ids[vec][0]==0) {
        if(vec==1)        { Nv++; return Nv; }
        else if(vec==2)   { Ne++; return Ne; }
        else if(vec==3)   { Nc++; return Nc; }
    }
    ids[vec][0]--; return ids[vec][ids[vec][0]+1];
}
//****************************************************************************
/*
 * freeID: Implement the free id operation for the C vertex-model code.
 * Parameters: int vec, int i.
 * Returns: see the C signature; most routines update global vertex-model state.
 */
void freeID(int vec, int i){
    ids[vec][0]++; ids[vec][ids[vec][0]]=i;
}
//****************************************************************************
