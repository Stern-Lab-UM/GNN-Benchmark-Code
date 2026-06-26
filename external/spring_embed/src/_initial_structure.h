//****************************************************************************
//****************************************************************************
//***********************INITIAL STRUCTURE************************************
//****************************************************************************
//****************************************************************************
void read_simulation_block(const char *filename, const char *target_sim_id) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("File open failed");
        return;
    }

    char line[LINE_SIZE];
    int found_sim = 0;

    while (fgets(line, sizeof(line), fp)) {
        // Check for "Simulation id:" lines
        if (strncmp(line, "Simulation id:", 14) == 0) {
            char current_sim_id[LINE_SIZE];
            sscanf(line, "Simulation id: %s", current_sim_id);

            if (strcmp(current_sim_id, target_sim_id) == 0) {
                found_sim = 1;
                printf("Found simulation: %s\n", current_sim_id);
            } else {
                // If already reading and hit a new simulation, stop
                if (found_sim) break;
                found_sim = 0;
            }
            continue;
        }

        // If inside the target simulation block, process data lines
        if (found_sim) {
            int cell1, cell2, was_flipped;
            double in_pref, out_pref, pred_len;

            int n = sscanf(line, "%d %d %lf %d %lf %lf", &cell1, &cell2, &in_pref, &was_flipped, &out_pref, &pred_len);

            if (n == 6) {
//                printf("Data: %d %d %.6f %d %.6f %.6f\n", cell1, cell2, in_pref, was_flipped, out_pref, pred_len);
                int omg=1;
                for(int i=1; i<=Ne; i++) if(exist[2][i]==1){
                    if( (e_cells[i][1]==cell1 && e_cells[i][2]==cell2) || (e_cells[i][1]==cell2 && e_cells[i][2]==cell1) ){
                        e_inLen[i]=in_pref;
                        e_outLen[i]=out_pref;
                        e_wasFlipped[i]=was_flipped;
                        e_l0[i]=pred_len;
                        if(e_wasFlipped[i]>0.5) printf("%d  %d  %d  %g  %d  %g  %g\n", i, cell1, cell2, e_inLen[i], e_wasFlipped[i], e_outLen[i], e_l0[i]);
                        omg=0;
                    }
                }
                if(omg==1){
                    e_inLen[0]=in_pref;
                    e_outLen[0]=out_pref;
                    e_wasFlipped[0]=was_flipped;
                    e_l0[0]=pred_len;
                    printf("-->%d  %d  %g  %d  %g  %g\n", cell1, cell2, e_inLen[0], e_wasFlipped[0], e_outLen[0], e_l0[0]);
                }
            }
        }
    }

    fclose(fp);
}
//****************************************************************************
void set_initial_fromFile(const char *filename2, const char *filename, const char *target_sim_id){

    int *sides; sides = new int[16];
    int nrSides;

    //FILE
    FILE *file1; file1 = fopen(filename2, "rt");


    //V,E,P,C
    int nrV=0, nrE=0, nrC=0;
    fscanf(file1, "%d  %d  %d\n", &nrV, &nrE, &nrC);

    //perioXYZ
    fscanf(file1, "%lf  %lf\n", &perioXYZ[1], &perioXYZ[2]);

    //VERTICES
    double xx, yy;
    for(int i=1; i<=nrV; i++){
        fscanf(file1, "%lf  %lf\n", &xx, &yy);
        make_vertex(xx,yy);
    }

    //EDGES
    int v1, v2;
    for(int i=1; i<=nrE; i++){
        fscanf(file1, "%d  %d\n", &v1, &v2);
        make_edge(v1,v2,0);
    }

    //POLYGONS
    //  2026-06-02 FIX: a cell line is "count s1 s2 ... s_count <zero padding>".
    //  The zero-padding WIDTH has varied across vt2d generations (12 vs 15 slots),
    //  and the old fixed-field fscanf (count + 12) desyncs on the wider padding --
    //  it reads a padding 0 as an edge id -> "element id=0" abort. Read nrSides,
    //  then exactly nrSides edge ids, then skip to end-of-line. Robust to any
    //  padding width and any cell size (<=15).
    for(int i=1; i<=nrC; i++){
        fscanf(file1, "%d", &nrSides);
        for(int k=1; k<=15; k++) sides[k]=0;
        for(int k=1; k<=nrSides && k<=15; k++) fscanf(file1, "%d", &sides[k]);
        int ch; while((ch=fgetc(file1))!='\n' && ch!=EOF);
        cell_sides(nrSides,sides,0);
    }
    delete [] sides;

    //MAKE CELLS
    for(int i=1; i<=Nc; i++) if(c[i][0]!=0) make_cell(i);

    fclose(file1);

    read_simulation_block(filename, target_sim_id);
}
//****************************************************************************
