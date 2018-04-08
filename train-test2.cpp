#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace cv;
using namespace cv::ml;
using namespace std;
int main(int argc, char** argv)
{
    //--------------准备训练数据--------------------
    //定义相关变量
    int ImgWidht = 40;
    int ImgHeight = 40;
    vector<string> img_path;
    vector<int> img_catg;
    int nLine = 0;
    string buf;
    int lab;
    ifstream svm_data("SVM_DATA.txt");
    unsigned long n;
    //依次读入样本的名和标签
    while(svm_data)
    {
        if(getline(svm_data,buf))          //读取一行内容到buf变量
        {
            if(nLine<10)
            {
                img_catg.push_back(0);     //图像类别，前10为数字0
                img_path.push_back(buf);    //图像路径
            }
            if(nLine>=10&&nLine<20)
            {
                img_catg.push_back(1);     //图像类别11-20为数字1
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=20&&nLine<30)
            {
                img_catg.push_back(2);     //图像类别11-20为数字2
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=30&&nLine<40)
            {
                img_catg.push_back(3);     //图像类别11-20为数字3
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=40&&nLine<50)
            {
                img_catg.push_back(4);     //图像类别11-20为数字4
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=50&&nLine<60)
            {
                img_catg.push_back(5);     //图像类别11-20为数字5
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=60&&nLine<70)
            {
                img_catg.push_back(6);     //图像类别11-20为数字6
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=70&&nLine<80)
            {
                img_catg.push_back(7);     //图像类别11-20为数字7
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=80&&nLine<90)
            {
                img_catg.push_back(8);     //图像类别11-20为数字8
                img_path.push_back(buf);   //图像路径
            }
            if(nLine>=90&&nLine<100)
            {
                img_catg.push_back(9);     //图像类别11-20为数字9
                img_path.push_back(buf);   //图像路径
            }
            nLine++;
        }
    }
    svm_data.close();//关闭文件
    //定义相关变量
    Mat data_mat, res_mat;
    int nImgNum =nLine;           //读入样本数量
    res_mat = Mat::zeros(nImgNum, 1, CV_32SC1);
    Mat src;
    Mat trainImg = Mat::zeros(ImgHeight, ImgWidht, CV_8UC3);  //需要分析的图片
    //依次提取样本的HOG特征
    for (string::size_type i = 0; i != img_path.size(); i++)
    {
        //读入样本
        src = imread(img_path[i].c_str(), 1);  //依次读入样本图片
        cout << " processing " << img_path[i].c_str() << endl;
        resize(src, trainImg, cv::Size(ImgWidht, ImgHeight), 0, 0, INTER_CUBIC);   //将样本图片调整为标准大小

        //对样本提取HOG特征向量
        HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWidht, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2
        vector<float> descriptors;  //存放提取HOG特征向量的结果数组
        hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //对读入的样本图片提取HOG特征向量
        if (i == 0)
        {
            data_mat = Mat::zeros(nImgNum, descriptors.size(), CV_32FC1); //根据输入图片大小进行分配空间
        }
        cout << "HOG dims: " << descriptors.size() << endl;
        //存储所提取的样本的HOG特征值
        n = 0;
        for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
        {
            data_mat.at<float>(i, n) = *iter;
            n++;
        }

        //存储样本的标签
        res_mat.at<int>(i, 0) = img_catg[i];
        cout << " end processing " << img_path[i].c_str() << " " << img_catg[i] << endl;
    }
//    cout<<"res_mat: "<<endl<<res_mat<<endl;


    //----------------------------设置SVM参数---------------------------------
    cout << "\n------------------------------------------------------------\n";
    cout << "Starting training process" << endl;
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(10);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));

    //-------------------------------训练SVM------------------------------------------
    svm->train(data_mat, ROW_SAMPLE, res_mat);   //注意此处data_mat和res_mat的类型
    //利用训练数据和确定的学习参数,进行SVM学习
    svm->save("SVM_DATA.xml");
    cout << "Finished training process" << endl;
    //--------------------------------检测样本---------------------------------------

    //依次读入测试样本
    Mat test;
    test = imread("8_2.png", 1);   //读入图像
    resize(test, trainImg, cv::Size(ImgWidht, ImgHeight), 0, 0, INTER_CUBIC);   //要搞成同样的大小才可以检测到

    //提取测试样本的HOG特征
    HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWidht, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
    vector<float>descriptors;//结果数组
    hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //调用计算函数开始计算
    cout << "The Detection Result:" << endl;
    cout << "HOG dims: " << descriptors.size() << endl;
    //存储所提取的特征
    Mat SVMtrainMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
    n = 0;
    for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
    {
        SVMtrainMat.at<float>(0, n) = *iter;
        n++;
    }
    //对提取的特征进行检验
    int ret = svm->predict(SVMtrainMat);
    cout<<"ret: "<<ret<<endl;

    return 0;
}

