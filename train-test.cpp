#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>

using namespace std;
using namespace cv;

int main()
{
    int ImgWidth=40;
    int ImgHeight=40;        //图像长和宽都为40
    vector<string>img_path;  //训练图片读取路径
    vector<int>img_catg;     //训练图片标签
    int nLine=1;             //读取行数从1开始
    string buf;

    ifstream svm_data;
    svm_data.open("SVM_DATA.txt");
    unsigned long n;

    while(svm_data)
    {
        if(getline(svm_data,buf))          //读取一行内容到buf变量
        {
            if(nLine<=10)
            {
                img_catg.push_back(0);     //图像类别，前10为数字1
                img_path.push_back(buf);    //图像路径
            }
            if(nLine>=11&&nLine<=20)
            {
                img_catg.push_back(1);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=21&&nLine<=30)
            {
                img_catg.push_back(2);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=31&&nLine<=40)
            {
                img_catg.push_back(3);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=41&&nLine<=50)
            {
                img_catg.push_back(4);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=51&&nLine<=60)
            {
                img_catg.push_back(5);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=61&&nLine<=70)
            {
                img_catg.push_back(6);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=71&&nLine<=80)
            {
                img_catg.push_back(7);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=81&&nLine<=90)
            {
                img_catg.push_back(8);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=91&&nLine<=100)
            {
                img_catg.push_back(9);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            nLine++;
        }
    }
    svm_data.close();                        //读取完成之后关闭文件

    Mat data_mat,res_mat;                    //mat变量存储图片数据和标签
    int nImgNum=nLine-1;                     //读取的总样本数量
    cout<<"nImgNum: "<<nImgNum<<endl;

    //HOG特征数计算：9×[(16/8)*(16/8)]*{[(40-16)/8+1]*[(40-16)/8+1]}=576
    data_mat=Mat::zeros(nImgNum,576,CV_32FC1); //存储样本的矩阵，行数为样本数量，列数为样本的特征向数
    res_mat=Mat::zeros(nImgNum,1,CV_32SC1);    //标签矩阵，存储每个数字对应打类型

    Mat src;
    Mat trainImg=Mat::zeros(ImgHeight,ImgWidth,CV_8UC3);
    for(string::size_type i=0; i!=img_path.size(); i++)  //依次读取每一张图片
    {
        src=imread(img_path[i].c_str(),1);
        resize(src,trainImg,Size(40,40));
        cout<<"processing "<<img_path[i].c_str()<<endl;
        HOGDescriptor *hog=new HOGDescriptor(Size(ImgWidth,ImgHeight),Size(16,16),Size(8,8),Size(8,8),9); //构造HOG函数
        vector<float>descriptors;  //结果数组
        hog->compute(trainImg,descriptors,Size(1,1), Size(0,0));

        if(i==0)
        {
            data_mat=Mat::zeros(nImgNum,descriptors.size(),CV_32FC1);  //根据真实的特征向量数调整列数
        }
        cout<<"HOG dims: "<<descriptors.size()<<endl;    //特征向量数确实为576
        n=0;                                            //将一张图片的特征向量导入data_mat
        for(vector<float>::iterator iter=descriptors.begin(); iter!=descriptors.end(); iter++)
        {
            data_mat.at<float>(i,n)=*iter;
            n++;
        }
        //namedWindow("data_mat",WINDOW_NORMAL);
        //imshow("data_mat",data_mat);
        res_mat.at<float>(i,0)=img_catg[i];
        cout<<"img_path: "<<img_path[i]<<"\t"<<img_catg[i]<<endl;
        //waitKey();
    }
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setC(10);
    svm->setKernel(ml::SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
    svm->train(data_mat,ml::ROW_SAMPLE, res_mat);
    svm->save("SVM_DATA.xml");

    //下面进行SVM测试
    Mat testImg=imread("2.png",1);
    resize(testImg,trainImg,Size(40,40));
    HOGDescriptor *hog2=new HOGDescriptor(Size(ImgWidth,ImgHeight),Size(16,16),Size(8,8),Size(8,8),9);
    vector<float>descriptors2;
    hog2->compute(trainImg,descriptors2,Size(1,1), Size(0,0));    //调用计算函数开始计算
    Mat SVMtestMat=Mat::zeros(1,descriptors2.size(),CV_32FC1);

    n=0;
    for(vector<float>::iterator iter2=descriptors2.begin(); iter2!=descriptors2.end(); iter2++)
    {
        SVMtestMat.at<float>(0,n)=*iter2;
        n++;
    }

    int ret=svm->predict(SVMtestMat);
    printf("%d\n",ret);
    return 0;
}




