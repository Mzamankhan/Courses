#include <iostream>
#include <bits/stdc++.h>
#include <iostream>

#include <string>
#include <cstdlib>
#include <sstream>


using namespace std;

void check(string s);
ofstream write;
int count1=0;


int main()
{
    ifstream dataread;
    string firstline,like,reply,rest;
    int in_ann,in_bully;
    dataread.open("quads before.txt");
    write.open("quads.txt");
    getline(dataread,firstline);
    count1++;
    while(!dataread.eof())
    {
        getline(dataread,like,'\t');
        check(like);
        getline(dataread,reply,'\t');
        check(reply);
        getline(dataread,rest);
        cout<<"\nchecking rest"<<rest<<endl;
        write<<rest<<'\n';
        count1++;
        cout<<"count "<<count1<<endl;

    }

}

void check(string s)
{
    int len=s.length();
    cout<<s<<" length ";
    cout<<len<<endl;
    if(s[len-1]=='k')
    {
        for(int i=0;i<=len-2;i++)
        {
            if(s[i]=='.')
            {
                continue;
            }
            cout<<s[i];
            write<<s[i];
        }
        cout<<"000";
        write<<"000\t";

    }
    else
    {
        write<<s<<'\t';
    }


}
