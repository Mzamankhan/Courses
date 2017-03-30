#include <bits/stdc++.h>

using namespace std;

int *cost;
double *rel;
int **nums_memoized,**nums_bottomup;
double **BN_bup,**BN_memoized;
int budget,N;
int loc_used = 0;


void init()
{
	cout<<"Budget: "<<budget<<"\n";
	cout<<"Number of machines: "<<N<<"\n";

	BN_memoized = new double*[budget+1];
	nums_memoized = new int*[budget+1];

	for(int i=0;i<budget+1;i++)
	{
		BN_memoized[i] = new double[N+1];
		nums_memoized[i] = new int[N+1];
	}

	

	//fill out BN_memoized to -1
	for(int i=0;i<=budget;i++)
	{
		for(int j=0;j<=N;j++)
		{
			BN_memoized[i][j] = -1;
			nums_memoized[i][j] = 0;
		}
	}

	BN_bup = new double*[N+1];
	nums_bottomup = new int*[N+1];
	for(int i=0;i<=N;i++)
	{
		BN_bup[i] = new double[budget+1];
		nums_bottomup[i] = new int[budget+1];
	}



}

double calculate_rel(int type,int m)
{
	double oneminus = 1-rel[type];
	double power = pow(oneminus,m);
	double result = 1 - power;
	return result;

}

double memoized(int type, int b)
{
	//check if value already calculated
	if (BN_memoized[b][type] != -1) { return BN_memoized[b][type]; }
	else if(b<0) { return 0; }
	else if(b==0 && type>0) { return 0;}
	else if(b>=0 && type==0 ) { return 1; }
	else
	{
		//int index = ceil(b/cost[type]); // b/ci
		//double newarr[index] = {0}; // array to hold the reliabilitye from m=1 to m=b/ci
		int limit = floor(b/cost[type]);
		//cout<<"limit "<<limit<<" ";
		double max = -1000000000000000;
		int max_index = 0;
		for(int m=1;m<=limit;m++)
		{
			double temp_rel = memoized(type-1, b-(m*cost[type]))*calculate_rel(type,m);
			if(temp_rel>max)
			{
				max = temp_rel;
				//nums_memoized[type] = m;
				max_index = m;
			}
		}
		BN_memoized[b][type] = max;
		nums_memoized[b][type] = max_index;
		loc_used+=1;
		//cout<<"For type "<<type<<" with cost"<<cost[type]<<" reliability "<<rel[type]<<" Budget "<<b<<" max rel is "<<max<<"\n";

	}
}

void iterative(int type, int b)
{
	//fill out base cases
	for(int i=0;i<=N;i++)
	{
		BN_bup[0][i] = 0.0;
		BN_bup[i][0] = 0.0;
	}

	for(int i=1;i<=N;i++)
	{
		for(int b=0;b<budget;b++)
		{
			if(b<cost[i]) { BN_bup[i][b] = 0.0; }
		}
	}
	//bottom up approach

	for(int i=1;i<=N;i++)
	{
		for(int b=1;b<=budget;b++)
		{
			//no previous value calculated
			if (i==1)
			{
				int limit = floor(b/cost[i]);
				//cout<<"limit "<<limit<<" ";
				double max = -1000000000000000;
				int max_index = 0;
				for(int m=1;m<=limit;m++)
				{
					double c_rel = calculate_rel(i,m);
					if(c_rel>max)
					{
						max = c_rel;
						max_index = m;

					}
				}
				BN_bup[i][b] = max;	
				nums_bottomup[i][b] = max_index;
			}

			else
			{
				int limit = floor(b/cost[i]);
				//cout<<"limit "<<limit<<" ";
				double max = -1000000000000000;
				int max_index = 0;
				for(int m=1;m<=limit;m++)
				{
					double c_rel = calculate_rel(i,m);
					double prev_rel = BN_bup[i-1][b-(m*cost[i])];
					double new_rel = c_rel*prev_rel;
					if(new_rel>max)
					{
						max = new_rel;
						//num_bup[i] = m;
						max_index = m;
					}
				}
				BN_bup[i][b] = max;	
				nums_bottomup[i][b] = max_index;
			}
		}
	}



}

int main()
{
	//Read data
	cin>>budget;
	cin>>N;

	cost = new int[N+1];
	rel = new double[N+1];

	
	string n,n1;
	for(int i=1;i<=N;i++)
	{
		cin>>cost[i];
		cin>>rel[i];
		//cout<<cost[i]<<" "<<rel[i]<<"\n";
		
	}

	init(); //declare and initialize arrays

	//run memoized version
	double final_reliability = memoized(N, budget);
	cout<<"\nMemoized Version\nMaximum reliability "<<setprecision(16) <<final_reliability<<"\n"; //get and print reliability

	//print num of machines
	int cB = budget;
	for (int i=N;i>0;i--)
	{
		cout<<nums_memoized[cB][i]<<" copies of machine "<<i<<" of cost "<<cost[i]<<"\n";
		cB -= nums_memoized[cB][i] * cost[i];
	}

	//Memoization statistics
	double total = budget*N;
	double percentage = (loc_used/total)*100;

	cout<<"\nTotal Location "<<total<<"\n";
	cout<<"Location used: "<<loc_used<<"\n";
	cout<<"Percentage used: "<<percentage<<"\n\n";

	//run iterative version

	iterative(1,budget);
	double bup_reliability = BN_bup[N][budget];
	cout<<"\nIterative Version\nMaximum reliability "<<setprecision(16)<<bup_reliability<<"\n";

	int bB = budget;
	for (int i=N;i>0;i--)
	{
		cout<<nums_bottomup[i][bB]<<" copies of machine "<<i<<" of cost "<<cost[i]<<"\n";
		bB -= nums_bottomup[i][bB] * cost[i];
	}




	delete[] cost;
	delete[] rel;
	delete[] nums_memoized;
	delete[] BN_memoized;
	delete[] BN_bup;
	//delete[] nums_bottomup;

	return 0;
}

