#include <iostream>
#include <bits/stdc++.h>
#include <iostream>

#include <string>
#include <cstdlib>
#include <sstream>


using namespace std;

int main()
{
    ifstream dataread;
    dataread.open("quads.txt");
    string firstline,like,reply,rest;
    getline(dataread,firstline);
    double dlikes,dreplies;
    double like_sum=0.0,reply_sum=0.0;
    double count1=0.0;
    while(!dataread.eof())
    {
        getline(dataread,like,'\t');
        istringstream convert(like);
        convert>>dlikes;
        cout<<dlikes<<" ";
        like_sum=like_sum+dlikes;

        getline(dataread,reply,'\t');
        istringstream convert1(reply);
        convert1>>dreplies;
        cout<<dreplies<<'\n';
        reply_sum=reply_sum+dreplies;
        count1++;
        cout<<"count "<<count1<<endl;
        getline(dataread,rest);

    }
    cout<<"total likes "<<like_sum<<endl;
    cout<<"total replies "<<reply_sum<<endl;
    cout<<"count "<<count1<<endl;
    double like_mean=like_sum/count1;
    double reply_mean=reply_sum/count1;
    ofstream write;
    write.open("mean.txt");
    write<<"mean likes\t"<<like_mean<<endl;
    write<<"mean replies\t"<<reply_mean<<endl;


}
