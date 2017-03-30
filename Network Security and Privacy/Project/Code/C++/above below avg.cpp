#include <iostream>
#include <bits/stdc++.h>
#include <iostream>

#include <string>
#include <cstdlib>
#include <sstream>


using namespace std;

int main()
{
    ifstream dataread,readmean;

    dataread.open("quads.txt");
    readmean.open("mean.txt");
    string firstline,like,reply,ann,bully,meantext,meanreply,meanlike;
    getline(dataread,firstline);
    double dlikes,dreplies,likemean,replymean,dann,dbully;

    //read and convert mean
    getline(readmean,meantext,'\t');getline(readmean,meanlike);
    getline(readmean,meantext,'\t');getline(readmean,meanreply);

    istringstream convertlike(meanlike);
    convertlike>>likemean;
    istringstream convertreply(meanreply);
    convertreply>>replymean;

    cout<<"means like, reply "<<likemean<<" "<<replymean<<endl;

    /////////////////////////////////////////create ofstreams
    ofstream likeabove,likebelow,replyabove,replybelow;
    likeabove.open("likeabove.txt");
    likebelow.open("likebelow.txt");
    replyabove.open("replyabove.txt");
    replybelow.open("replybelow.txt");
    ///////////////////////////////////////////mean variables
    double annabove_like=0.0,annbelow_like=0.0,rannabove=0.0,rannbelow=0.0;
    double bullyabove_like=0.0,bullybelow_like=0.0,rbullyabove=0.0,rbullybelow=0.0;
    double annabove_like_count=0.0,annbelow_like_count=0.0,reply_annabove_count=0.0,reply_annbelow_count=0.0;
    double bullybove_like_count=0.0,bullybelow_like_count=0.0,reply_bullyabove_count=0.0,reply_bullybelow_count=0.0;


    double count1=0.0;
    while(!dataread.eof())
    {
        getline(dataread,like,'\t');
        istringstream convert(like);
        convert>>dlikes;
        cout<<dlikes<<" ";


        getline(dataread,reply,'\t');
        istringstream convert1(reply);
        convert1>>dreplies;
        cout<<dreplies<<" ";

        count1++;
        cout<<"count "<<count1<<endl;
        getline(dataread,ann,'\t');
        istringstream convertann(ann);
        convertann>>dann;

        getline(dataread,bully);
        istringstream convertbully(bully);
        convertbully>>dbully;

        if(dlikes>=likemean)
        {
            likeabove<<like<<'\t'<<reply<<'\t'<<ann<<'\t'<<bully<<'\n';
            annabove_like=annabove_like+dann;
            annabove_like_count++;

            bullyabove_like=bullyabove_like+(dbully+5.0);
            bullybove_like_count++;

        }
        else
        {
            likebelow<<like<<'\t'<<reply<<'\t'<<ann<<'\t'<<bully<<'\n';
            annbelow_like=annbelow_like+dann;
            annbelow_like_count++;

            bullybelow_like=bullybelow_like+(dbully+5.0);
            bullybelow_like_count++;

        }

        if(dreplies>=replymean)
        {
            replyabove<<like<<'\t'<<reply<<'\t'<<ann<<'\t'<<bully<<'\n';
            rannabove=rannabove+dann;
            reply_annabove_count++;

            rbullyabove=rbullyabove+(dbully+5.0);
            reply_bullyabove_count++;
        }
        else
        {
            replybelow<<like<<'\t'<<reply<<'\t'<<ann<<'\t'<<bully<<'\n';
            rannbelow=rannbelow+dann;
            reply_annbelow_count++;

            rbullybelow=rbullybelow+(dbully+5.0);
            reply_bullybelow_count++;

        }


    }

  double above_ann_like=annabove_like/annabove_like_count;
  double above_bully_like=bullyabove_like/bullybove_like_count;

  double below_ann_like=annbelow_like/annbelow_like_count;
  double below_bully_like=bullybelow_like/bullybelow_like_count;

  double reply_ann_above=rannabove/reply_annabove_count;
  double reply_bully_above=rbullyabove/reply_bullyabove_count;

  double reply_ann_below=rannbelow/reply_annbelow_count;
  double reply_bully_below=rbullybelow/reply_bullybelow_count;

  likeabove<<"above_ann_like "<<above_ann_like<<endl;
  likeabove<<"above_bully_like "<<above_bully_like<<endl;
  likebelow<<"below_ann_like "<<below_ann_like<<endl;
  likebelow<<"below_bully_like "<<below_bully_like<<endl;
  replyabove<<"reply_ann_above "<<reply_ann_above<<endl;
  replyabove<<"reply_bully_above "<<reply_bully_above<<endl;
  replybelow<<"reply_ann_below "<<reply_ann_below<<endl;
  replybelow<<"reply_bully_below "<<reply_bully_below<<endl;
  cout<<"\n count at end is "<<count1;
}
