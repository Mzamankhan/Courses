#include <iostream>
#include <bits/stdc++.h>
#include <iostream>

#include <string>
#include <cstdlib>
#include <sstream>


using namespace std;

int main()
{
    int ann_array[11]={0};
    int bully_array[11]={0};
    double ann_percent[11]={0},bully_percent[11]={0};
    for(int i=0;i<11;i++)
    {
        ann_array[i]=0;
        bully_array[i]=0;
    }
    ifstream dataread;
    string firstline,like,reply,ann,bully;
    int in_ann,in_bully;
    dataread.open("quads.txt");
    getline(dataread,firstline);
    double count1=0;
    while(!dataread.eof())
    {
        getline(dataread,like,'\t');
        getline(dataread,reply,'\t');
        getline(dataread,ann,'\t');
        istringstream convert(ann);
        convert>>in_ann;
       // cout<<"ann"<<ann<<" "<<in_ann<<" ";
        ann_array[in_ann]++;
        //cout<<ann_array[in_ann]<<"\n";

        getline(dataread,bully);
        istringstream convert1(bully);
        convert1>>in_bully;
        //cout<<"bully"<<bully<<" "<<in_bully<<" ";
        bully_array[in_bully+5]++;
        //cout<<bully_array[in_bully+5]<<"\n";

       // cout<<like<<" "<<reply<<" "<<ann<<" "<<bully<<"\n";
       // cout<<in_ann<<" "<<in_bully<<"\n";
       count1++;
    }
    cout<<count1<<"\n";

   // int sum=0;
    for(int i=0;i<11;i++)
    {
        //cout<<ann_array[i]<<" ";
        //sum=sum+ann_array[i];
        ann_percent[i]=(ann_array[i]/count1)*100;
        bully_percent[i]=(bully_array[i]/count1)*100.0;

    }
    //cout<<sum<<"\n";

   //  int sum1=0;
   cout<<"i ann bully\n";
   ofstream write;
   write.open("percentage.txt");
    for(int i=0;i<11;i++)
    {
      //  cout<<bully_array[i]<<" ";
        //sum1=sum1+bully_array[i];
        cout<<i<<" "<<ann_percent[i]<<" "<<bully_percent[i]<<"\n";
        write<<i<<" "<<ann_percent[i]<<" "<<bully_percent[i]<<"\n";

    }
    //cout<<sum1;

}
