
//***********************************PARALLEL K-MEANS CLUSTERING (USING OMP)**************************************

/* [numClusters]: no. objects assigned in each new cluster */
/*delta :  % of objects change their clusters */
/* **clusters: out: [numClusters][dimensions] */
/* **newClusters: [numClusters][dimensions] */


#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include<math.h>
#include <omp.h>
int     _debug;
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


float** file_read(char *filename,int  *numObjs,int  *dimensions){

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
        (*dimensions) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                while (strtok(NULL, " ,\t\n") != NULL) (*dimensions)++;
                break;
            }
        }

        rewind(infile);
        printf("File %s numObjs   = %d\n",filename,*numObjs);
        printf("File %s dimensions = %d\n",filename,*dimensions);

        len = (*numObjs) * (*dimensions);

        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);

        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);

        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*dimensions);

        i = 0;

        while (fgets(line, lineLen, infile) != NULL) {

            if (strtok(line, " \t\n") == NULL)
              continue;

            for (j=0; j<(*dimensions); j++)
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


__inline static int find_nearest_cluster(int numClusters, int dimensions, float *object, float **clusters)
{
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


float** omp_kmeans(float **objects, int dimensions,int numObjs,int numClusters,float threshold,int *membership,int *loop_iterations)
{

    int i, j, k, index, loop=0;
    int  *newClusterSize;
    float  delta;
    float  **clusters;
    float  **newClusters;
    double  timing;
    int nthreads;
    int **local_newClusterSize;
    float ***local_newClusters;

    nthreads = omp_get_max_threads();

    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    assert(clusters != NULL);

    clusters[0] = (float*)  malloc(numClusters * dimensions * sizeof(float));
    assert(clusters[0] != NULL);

    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + dimensions;

    for (i=0,index=0; index<numClusters; i+= numObjs/numClusters-1,index++)
        for (j=0; j<dimensions; j++)
            clusters[index][j] = objects[i][j];

    for (i=0; i<numObjs; i++) membership[i] = -1;

    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * dimensions, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + dimensions;

    local_newClusterSize    = (int**) malloc(nthreads * sizeof(int*));
    assert(local_newClusterSize != NULL);
    local_newClusterSize[0] = (int*)  calloc(nthreads*numClusters,
                                             sizeof(int));
    assert(local_newClusterSize[0] != NULL);
    for (i=1; i<nthreads; i++)
        local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;

        //**3D Array
    local_newClusters    =(float***)malloc(nthreads * sizeof(float**));
    assert(local_newClusters != NULL);
    local_newClusters[0] =(float**) malloc(nthreads * numClusters *
                                           sizeof(float*));
    assert(local_newClusters[0] != NULL);
    for (i=1; i<nthreads; i++)
        local_newClusters[i] = local_newClusters[i-1] + numClusters;
    for (i=0; i<nthreads; i++) {
        for (j=0; j<numClusters; j++) {
            local_newClusters[i][j] = (float*)calloc(dimensions,
                                                     sizeof(float));
            assert(local_newClusters[i][j] != NULL);
        }
    }


    if (_debug) timing = omp_get_wtime();
    do {
        delta = 0.0;

          #pragma omp parallel \
                  shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
          {
              int tid = omp_get_thread_num();
              #pragma omp for \
                          private(i,j,index) \
                          firstprivate(numObjs,numClusters,dimensions) \
                          schedule(static) \
                          reduction(+:delta)
              for (i=0; i<numObjs; i++) {

                  index = find_nearest_cluster(numClusters, dimensions, objects[i], clusters);

                  if (membership[i] != index) delta += 1.0;

                  membership[i] = index;

                  local_newClusterSize[tid][index]++;
                  for (j=0; j<dimensions; j++)
                      local_newClusters[tid][index][j] += objects[i][j];
              }
          } /* end of #pragma omp parallel */

          // Main thread performs the array reduction
          for (i=0; i<numClusters; i++) {
              for (j=0; j<nthreads; j++) {
                  newClusterSize[i] += local_newClusterSize[j][i];
                  local_newClusterSize[j][i] = 0.0;
                  for (k=0; k<dimensions; k++) {
                      newClusters[i][k] += local_newClusters[j][i][k];
                      local_newClusters[j][i][k] = 0.0;
                  }
              }
          }

      for (i=0; i<numClusters; i++) {
          for (j=0; j<dimensions; j++) {
              if (newClusterSize[i] > 1)
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
           double  timing, clustering_timing, total_timing;
           int     loop_iterations;


    // Default values
    numClusters       = 0;
    threshold         = 0.001;

    printf("Usage-> Arguments: filename  num_clusters\n");

    if(argc == 3){
       sscanf(argv[1],"%s",filename);
       sscanf(argv[2],"%d",&maxClusters);
    }

    objects = file_read( filename, &numObjs, &dimensions);

    if (objects == NULL) exit(1);

    printf("\nPerforming **** Regular Kmeans V4 for multiple k (Parallel Version using OpenMP)*******\n");
    printf("Number of threads = %d\n", omp_get_max_threads());
    printf("Input file:     %s\n", filename);
    printf("numObjs       = %d\n", numObjs);
    printf("dimensions     = %d\n", dimensions);
    printf("maxClusters   = %d\n", maxClusters);
    printf("threshold     = %.4f\n", threshold);

    membership = (int*) malloc(numObjs * sizeof(int));        /* membership: the cluster id for each data object */

    FILE *fptr;
    char  outFileName[1024];
    sprintf(outFileName, "%s.ave_dist_trend", filename);
    fptr = fopen(outFileName, "w");

    double average_distance;

    timing            = wtime();
    total_timing = timing;

    for(int i=2;i<maxClusters;i++){

        numClusters=i;
  //********** This section is for measuring the computation time and # of iteration for each k*******
        //
        // timing            = omp_get_wtime();
        // clustering_timing = timing;


        assert(membership != NULL);

        clusters = omp_kmeans(objects, dimensions, numObjs, numClusters, threshold, membership,&loop_iterations);


        // timing            = omp_get_wtime();
        // clustering_timing = timing - clustering_timing;

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

    printf("Total timing = %10.4f sec\n", total_timing);

    return(0);
}
