#include <iostream>
#include <bits/stdc++.h>
#include <iostream>

#include <string>
#include <cstdlib>
#include <sstream>
#include <ctime>

using namespace std;


int maxNodeId=0;
int node_count=0;
double **credit;

int old_credit=0;
int new_credit=1;
double write_duration=0;


vector<vector<int> > read_data(const char* infile);
void randomwalk(vector<vector<int> > ,int,const char* outfile);
void createOutputFile(vector<vector<int> >,const char* outfile);

int main(int argc, char *argv[])
{
    //read the data into data structure
    vector<vector<int> > edges = read_data(argv[1]);

    //create the output file and write the node number and degree
    createOutputFile(edges,argv[2]);

    //define number of rounds and run rounds
    int round=atoi(argv[3]);
    for(int loop=0;loop<round;loop++)
    {
        randomwalk(edges, loop,argv[2]);
    }

    printf("Time to write is %f\n",write_duration);

    free(credit);
    return 1;
}

vector<vector<int> > read_data(const char* infile)
{
    FILE * pFILE;
    int first,second;
    pFILE=fopen(infile,"r");

    clock_t start;
    double read_duration;

    start = clock();

    //find the the size of the vector to create
    if(pFILE!=NULL)
    {
        while(fscanf(pFILE,"%u %u",&first,&second)==2)
        {
           if(first>maxNodeId)
           {
               maxNodeId=first;
           }
           if(second>maxNodeId)
           {
               maxNodeId=second;
           }
        }

    }

    fclose(pFILE);

    //create vector
    vector<vector<int> > nodes(maxNodeId);
   //create the two dimensional array to hold credit information
   credit=(double **)malloc(maxNodeId*sizeof(double));
   for(int i=0;i<maxNodeId;i++)
   {
       credit[i]=(double *)malloc(2*sizeof(double));
       credit[i][0]=1;
       credit[i][1]=0;
   }

    //read file into memory
    FILE * fileRead;
    fileRead=fopen(infile,"r");
    if(fileRead!=NULL)
    {
        while(fscanf(fileRead,"%u %u",&first,&second)==2)
        {
            //cout<<"first is "<<first<<" second is "<<second<<"\n";
            nodes[first-1].push_back(second);
            nodes[second-1].push_back(first);
        }

    }
     read_duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
     printf("Time to read was %f\n",read_duration);

    return nodes;

}



void randomwalk(vector<vector<int> >  edges,int round,const char* outfile)
{

    string line;
    ifstream read(outfile);

    ofstream write;
    write.open("temp.txt");

    clock_t start;
    double round_duration;

    start = clock();

    for(int i=1;i<=maxNodeId;i++)
    {
        //check if neighbors exist
        credit[i-1][new_credit]=0;
        if(edges[i-1].size()>0)
        {
            double total_credit=0;
            for(int j=0;j<edges[i-1].size();j++)
            {
                int n_id=edges[i-1][j]-1;

                total_credit = total_credit+(credit[n_id][old_credit]/edges[n_id].size());
            }
            credit[i-1][new_credit]=total_credit;
        }
        //read the corresponding line from file
        clock_t write1;
        write1=clock();

        getline(read,line);
        if (read.bad())
            {
                // IO error
               printf("error\n");
            }
        //write to temp file
        write<<line<<"\t"<<setprecision(8)<<credit[i-1][new_credit]<<"\n";

        write_duration += (double)( clock() - write1 ) / (double) CLOCKS_PER_SEC;

    }
    //set old credit and reset new credit to 0
        int temp=old_credit;
        old_credit=new_credit;
        new_credit=temp;

     round_duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
     printf("Time of round  %d is %f\n",round,round_duration);

    read.close();
    write.close();
     //delete old output file
    int deleteold=remove( outfile);
    if(deleteold!=0)
    {
        printf("Error deleting");
    }
    //rename temp to output file
    int changename= rename( "temp.txt" , outfile);
    if(changename!=0)
    {
        printf("Error renaming");
    }


}

void createOutputFile(vector<vector<int> > edges,const char* outfile)
{
    ofstream write;
    FILE *out;
    out=fopen(outfile,"w+");

    clock_t start;
    start=clock();


    for(int i=1;i<=maxNodeId;i++)
    {
        //check if i is a valid node, should have a valid degree node
        if(edges[i-1].size()>0)
        {
            int s=edges[i-1].size();
            fprintf(out,"%d\t%d\n",i,s);

        }

    }
     fclose(out);
     write_duration += (double)( clock() - start ) / (double) CLOCKS_PER_SEC;


}




