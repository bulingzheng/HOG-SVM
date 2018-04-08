#include "functions.h"
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;

int HandNumber;       //识别出打结果以int形式展示
float Distance;       //离最近数字的距离

string to_string(int &x)
{
    stringstream ss;
    string y;
    ss<<x;
    ss>>y;
    return y;
}

int main()
{
    int output12Images[10];
    output12Images[0]=0;
    output12Images[1]=174;
    output12Images[2]=170;
    output12Images[3]=178;
    output12Images[4]=175;
    output12Images[5]=172;
    output12Images[6]=165;
    output12Images[7]=170;
    output12Images[8]=170;
    output12Images[9]=181;
    ofstream outfile, outfile2;
    outfile.open("output_all.txt");
    outfile2.open("output_error.txt");

    /////////初始化KNearest数字识别（手写体）
    FileStorage fsClassifications("classifications.xml",FileStorage::READ);  //打开xml标签文件
    if(!fsClassifications.isOpened())
    {
        cout<<"classifications.xml is not opened..."<<endl;
    }
    Mat matClassificationInts;
    fsClassifications["classifications"]>>matClassificationInts; //calssifications.xml文件中“classifications”标签之间打内容
    fsClassifications.release();                                 //关闭文件

    FileStorage fsTrainingImages("images.xml",FileStorage::READ); //打开训练图片xml
    if(!fsTrainingImages.isOpened())
    {
        cout<<"images.xml is not opened..."<<endl;
    }
    Mat matTrainingImagesAsFlattenedFloats;
    fsTrainingImages["images"]>>matTrainingImagesAsFlattenedFloats; //读取图片数据到Mat变量
    fsTrainingImages.release();

    //Ptr<ml::KNearest> model(ml::KNearest::create());  //实例化KNearest对象
    //model->train(matTrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, matClassificationInts); //训练KNN网络
    Ptr<ml::SVM> modelSVM(ml::SVM::create()); //实例化SVM对象
    modelSVM->train(matTrainingImagesAsFlattenedFloats,ml::ROW_SAMPLE,matClassificationInts); //训练SVM网络

    HOGDescriptor *hog=new HOGDescriptor(Size(40,40),Size(16,16),Size(8,8),Size(8,8),9);  //获取HOG特征的方式
    vector<float> descriptors;

    //此处开始正式识别
    for(int i=1; i<=9; i++)
    {
        for(int j=1; j<=output12Images[i]; j++)
        {
            string ImgName="output12imagemiddle50*50/"+to_string(i)+"/"+to_string(i)+"_"+to_string(j)+".png";
            Mat img=imread(ImgName,0);
            resize(img,img,Size(40,40));
            imshow("resize",img);

            hog->compute(img,descriptors);   //计算待识别图像的HOG特征
            Mat matDescriptors(1,(int)(descriptors.size()),CV_32FC1,descriptors.data());  //HOG特征保存在mat变量中
            Mat matCurrent(0,0,CV_32F);      //识别出打结果保存在mat变量中
            Mat m1(0,0,CV_32F);
            Mat m2(0,0,CV_32F);
           HandNumber=modelSVM->predict(descriptors);    //识别出的结果

            cout<<ImgName<<" ->HandNumber: "<<HandNumber<<endl;
            outfile<<ImgName<<" ->HandNumber: "<<HandNumber<<endl;
            if(HandNumber !=i)
            {
                outfile2<<ImgName<<" ->HandNumber: "<<HandNumber<<endl;
            }
            //waitKey(0);

        }
    }



    return 0;
}




