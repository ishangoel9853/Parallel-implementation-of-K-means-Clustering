
//***********************************PARALLEL K-MEANS CLUSTERING (USING OMP)**************************************

/* [numClusters]: no. objects assigned in each new cluster */
/*delta :  % of objects change their clusters */
/* **clusters: out: [numClusters][dimensions] */
/* **newClusters: [numClusters][dimensions] */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include<math.h>

#define MAX_CHAR_PER_LINE 128


double wtime(void){
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /* in seconds */
               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
    return now_time;
}


float** file_read(char *filename, int  *numObjs,    int  *numCoords)  {
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

        /* first find the nmber of objects */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            /* check each line to find the max line length */
            while (strlen(line) == lineLen-1) {
                /* this line read is not complete */
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                /* increase lineLen */
                lineLen += MAX_CHAR_PER_LINE;
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
                /* ignore the id (first coordiinate): numCoords = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break; /* this makes read from 1st object */
            }
        }
        rewind(infile);
        printf("File %s numObjs   = %d\n",filename,*numObjs);
        printf("File %s numCoords = %d\n",filename,*numCoords);

        /* allocate space for objects[][] and read all objects */
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;
        /* read all objects */
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++)
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }

        fclose(infile);
        free(line);

    return objects;
}


__inline static float euclid_dist_2(int dimensions, float *x1, float *x2){
    int i;
    float ans=0.0;
    for (i=0; i<dimensions; i++)
    {

        ans += (x1[i]-x2[i]) * (x1[i]-x2[i]);
    }

    return(ans);
}



__inline static int find_nearest_cluster(int numClusters, int dimensions, float *object, float **clusters){
    int   j, i;
    float min_d;

    float* dist = (float*) malloc(numClusters * sizeof(float));

    #pragma omp parallel for
    for (i=0; i<numClusters; i++) {
        dist[i] = euclid_dist_2(dimensions, object, clusters[i]);
    }

    min_d = dist[0];
    j=0;

    #pragma omp parallel for reduction(min:min_d)
    for (i=0; i<numClusters; i++) {
        if(min_d>dist[i])
        {
            min_d = dist[i] ;
        }
    }

   #pragma omp parallel for
    for (i=0; i<numClusters; i++) {
        if(min_d==dist[i])
        {
            j = i ;
        }
    }


    return(j);
}


/* return an array of cluster centers of size [numClusters][dimensions]       */
float** seq_kmeans(float **objects,      /* in: [numObjs][dimensions] */
                   int     dimensions,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations){
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **clusters;       /* out: [numClusters][dimensions] */
    float  **newClusters;    /* [numClusters][dimensions] */


    clusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(numClusters * dimensions * sizeof(float));
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + dimensions;


    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i=0,index=0; index<numClusters; i+= numObjs/numClusters-1,index++)
        for (j=0; j<dimensions; j++)
            clusters[index][j] = objects[i][j];

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;


    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    newClusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * dimensions, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + dimensions;


    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = find_nearest_cluster(numClusters, dimensions, objects[i],
                                         clusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<dimensions; j++)
                newClusters[index][j] += objects[i][j];
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<dimensions; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}


double ave_dist(float **objects,      /* in: [numObjs][dimensions] */
                   int     dimensions,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,
                   int    *membership,
                   float **clusters )
{

      double dist=0,temp=0;
      for(int i=0; i<numObjs; i++){
        temp=euclid_dist_2(dimensions,objects[i],clusters[membership[i]]);
        temp=sqrt(temp);
        dist+=temp;
      }
      double average_distance;
      average_distance=dist/numObjs;
      //printf("average distance: %10.8f ",average_distance);
      return average_distance;

}


int main(int argc, char **argv) {

    int     i, j;
    int     numClusters,maxClusters, dimensions, numObjs;
    int    *membership;    /* [numObjs] */
    char   filename[100];
    float **objects;       /* [numObjs][dimensions] data objects */
    float **clusters;      /* [numClusters][dimensions] cluster center */
    float   threshold;
    double  timing, clustering_timing,total_timing;
    int     loop_iterations;

    threshold        = 0.001;
    numClusters      = 0;

    char ch; char a;
    if(argc == 3){
       sscanf(argv[1],"%s",filename);
       sscanf(argv[2],"%d",&maxClusters);
    }

    if (strlen(filename) == 0 || maxClusters <= 1) {
        printf("Invalid Arguments\n");
        printf("Usage-> Arguments: filename  num_clusters\n");
        exit(-1);
    }

    objects = file_read(filename, &numObjs, &dimensions);
    if (objects == NULL) exit(1);

    printf("\nPerforming **** Regular Kmeans V2 (Parallel Version using OpenMP)*******\n");
    printf("Input file:     %s\n", filename);
    printf("numObjs       = %d\n", numObjs);
    printf("maxClusters   = %d\n", maxClusters);

    printf("dimensions     = %d\n", dimensions);
    printf("threshold     = %.4f\n", threshold);

    FILE *fptr;
    char  outFileName[1024];
    sprintf(outFileName, "%s.ave_dist_trend", filename);
    fptr = fopen(outFileName, "w");

    double average_distance;

    timing            = wtime();
    total_timing = timing;

    membership = (int*) malloc(numObjs * sizeof(int));

    for(int i=2;i<=maxClusters;i++){
        numClusters=i;
  //********** This section is for measuring the computation time and # of iteration for each k*******

      //  timing            = wtime();
      //  clustering_timing = timing;


        assert(membership != NULL);

        clusters = seq_kmeans(objects, dimensions, numObjs, numClusters, threshold, membership, &loop_iterations);

      //  timing            = wtime();
      //  clustering_timing = timing - clustering_timing;

        average_distance=ave_dist(objects,dimensions, numObjs, numClusters, membership, clusters);

        fprintf(fptr, "%d ", numClusters);
        fprintf(fptr, "%10.8f ", average_distance);
        fprintf(fptr, "\n");


        // printf("numClusters   = %d\t", numClusters);
        // printf("Loop iterations    = %d       ", loop_iterations);
        // printf("Computation timing = %10.8f sec\n", clustering_timing);

    }
    timing            = wtime();
    total_timing = timing - total_timing;

    fclose(fptr);
    free(objects[0]);
    free(objects);
    free(membership);
    free(clusters[0]);
    free(clusters);

    printf("Total Computation timing = %10.8f sec\n", total_timing);

    return(0);
}
