

//***********************************SEQUENTIAL K-MEANS CLUSTERING**************************************

/* [numClusters]: no. objects assigned in each new cluster */
/*delta :  % of objects change their clusters */
/* **clusters: out: [numClusters][dimentions] */
/* **newClusters: [numClusters][dimentions] */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#define MAX_CHAR_PER_LINE 128


double wtime(void)
{
    double now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +
               ((double)etstart.tv_usec) / 1000000.0;

    return now_time;
}


float** file_read(char *filename,int  *numObjs,int  *numCoords){

    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;

        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }

        /*finding the nmber of objects */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            while (strlen(line) == lineLen-1) {
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                lineLen += MAX_CHAR_PER_LINE;     /* increasing lineLen */

                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }

            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);

        //printf("lineLen = %d\n",lineLen);

        /* find the no. objects of each object */
        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break;
            }
        }

        rewind(infile);
        printf("File %s numObjs   = %d\n",filename,*numObjs);
        printf("File %s numCoords = %d\n",filename,*numCoords);

        len = (*numObjs) * (*numCoords);

        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);

        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);

        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;

        while (fgets(line, lineLen, infile) != NULL) {

            if (strtok(line, " \t\n") == NULL)
              continue;

            for (j=0; j<(*numCoords); j++)
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }

        fclose(infile);
        free(line);

    return objects;
}



int file_write(char *filename,int  numClusters,int numObjs,int numCoords,float  **clusters,int  *membership){
    FILE *fptr;
    int   i, j;
    char  outFileName[1024];

    sprintf(outFileName, "%s.output_cluster_centers", filename);
    printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",numClusters, outFileName);

    fptr = fopen(outFileName, "w");

    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);

        for (j=0; j<numCoords; j++)
            fprintf(fptr, "%f ", clusters[i][j]);

        fprintf(fptr, "\n");
    }
    fclose(fptr);


    sprintf(outFileName, "%s.output_membership", filename);
    printf("Writing membership of N=%d data objects to file \"%s\"\n",numObjs, outFileName);

    fptr = fopen(outFileName, "w");

    for (i=0; i<numObjs; i++)
        fprintf(fptr, "%d %d\n", i, membership[i]);

    fclose(fptr);

    return 1;
}



__inline static float euclidian_distance(int dimentions, float *x1, float *x2)
{
    int i;
    float ans=0.0;

    for (i=0; i<dimentions; i++)
        ans += (x1[i]-x2[i]) * (x1[i]-x2[i]);

    return(ans);
}


__inline static int find_nearest_cluster(int numClusters, int dimentions, float *object, float **clusters)
{
    int   j, i;
    float dist, min_d;

    min_d = euclidian_distance(dimentions, object, clusters[0]);
    j = 0;

    for (i=1; i<numClusters; i++) {

        dist = euclidian_distance(dimentions, object, clusters[i]);

        if (dist < min_d) {
            min_d = dist;
            j = i;
        }
    }
    return(j);
}


/* Return a 2D array of cluster centers of size [numClusters][dimentions]       */
float** seq_kmeans(float **objects, int  dimentions, int numObjs,int numClusters,float threshold,int *membership,int *loop_iterations){

    int  i, j, index, loop=0;
    int  *newClusterSize;
    float delta;
    float **clusters;
    float **newClusters;

    clusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);

    clusters[0] = (float*)  malloc(numClusters * dimentions * sizeof(float));
    assert(clusters[0] != NULL);

    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + dimentions;


      for (i=0; i<numClusters; i++)                       /* picking up first numClusters elements of objects[] as initial cluster centers*/
        for (j=0; j<dimentions; j++)
            clusters[i][j] = objects[i][j];


    for (i=0; i<numObjs; i++) membership[i] = -1;           /* initialize membership[] */


    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);

    newClusters[0] = (float*)  calloc(numClusters * dimentions, sizeof(float));
    assert(newClusters[0] != NULL);

    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + dimentions;

    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            index = find_nearest_cluster(numClusters, dimentions, objects[i], clusters);

            if (membership[i] != index)
              delta += 1.0;

            membership[i] = index;

            newClusterSize[index]++;
            for (j=0; j<dimentions; j++)
                newClusters[index][j] += objects[i][j];
        }

        for (i=0; i<numClusters; i++) {
            for (j=0; j<dimentions; j++) {

                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];

                newClusters[i][j] = 0.0;
            }
            newClusterSize[i] = 0;
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}


int main(int argc, char **argv) {

    int     i, j;
    int     numClusters, dimentions, numObjs;
    int    *membership;    /* [numObjs] */
    char   filename[100];
    float **objects;       /* [numObjs][dimentions] data objects */
    float **clusters;      /* [numClusters][dimentions] cluster center */
    float   threshold;
    double  timing, clustering_timing;
    int     loop_iterations;

    threshold        = 0.001;
    numClusters      = 0;

    printf("Usage-> Arguments: filename  num_clusters\n");

    char ch; char a;
    if(argc == 3){
       sscanf(argv[1],"%s",filename);
       sscanf(argv[2],"%d",&numClusters);
    }

    if (strlen(filename) == 0 || numClusters <= 1) {     printf("Invalid Arguments\n"); exit(-1); }

    objects = file_read(filename, &numObjs, &dimentions);
    if (objects == NULL) exit(1);

    timing            = wtime();
    clustering_timing = timing;

    membership = (int*) malloc(numObjs * sizeof(int));

    assert(membership != NULL);

    clusters = seq_kmeans(objects, dimentions, numObjs, numClusters, threshold, membership, &loop_iterations);

    free(objects[0]);
    free(objects);

    timing            = wtime();
    clustering_timing = timing - clustering_timing;


    file_write(filename, numClusters, numObjs, dimentions, clusters, membership);

    free(membership);
    free(clusters[0]);
    free(clusters);

    printf("\nPerforming **** Regular Kmeans (sequential version) ****\n");
    printf("Input file:     %s\n", filename);
    printf("numObjs       = %d\n", numObjs);
    printf("dimentions     = %d\n", dimentions);
    printf("numClusters   = %d\n", numClusters);
    printf("threshold     = %.4f\n", threshold);
    printf("Loop iterations    = %d\n", loop_iterations);
    printf("Computation timing = %10.8f sec\n", clustering_timing);

    return(0);
}
