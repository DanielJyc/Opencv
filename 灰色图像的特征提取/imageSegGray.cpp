#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
#include "featuresGLCM.h"
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>

using namespace std;
using namespace cv;

#define PI 3.1415926



/************************************************************************/
/*                 比较函数                                             */
/************************************************************************/
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}

/************************************************************************/
/*  获取面积最大的区域                                                                    */
/************************************************************************/
vector<Point> getMaxOfContours(vector<vector<Point>> &contours)
{
	std::sort(contours.begin(), contours.end(), compareContourAreas);  //对区域进行排序.compareContourAreas 为比较函数
	std::vector<cv::Point> biggestContour = contours[contours.size() - 1]; // 获取最大区域
	std::vector<cv::Point> smallestContour = contours[0]; // 获取最小区域
	cout << "最大区域" << contourArea(biggestContour) << endl;
	return biggestContour;
}

/************************************************************************/
/* 显示所有的火灾区域及其特征（周长和面积）                             */
/************************************************************************/
void getAllContoursFeatures(vector<vector<Point>> &contours)
{
	// 显示每个连通区域的大小
	std::cout << "Contours: " << contours.size() << std::endl;
	std::vector<vector<Point>>::const_iterator itContours = contours.begin();
	for (; itContours != contours.end(); ++itContours)
	{
		std::cout << "Size: " << itContours->size() << std::endl;
	}
	//显示每个区域的面积和周长
	for (int i = 0; i < contours.size(); i++)
	{
		printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));
	}
}

/************************************************************************/
/* 获取图片img的尖角个数                                               */
/************************************************************************/
int getCountOfFireConers(Mat img)
{
	//找出角点个数并标注在图中
	std::vector<cv::KeyPoint> keypoints; //特征点的向量
	keypoints.clear();
	FastFeatureDetector fast(250); //检测阈值 200 40
	fast.detect(img, keypoints); //进行检测
	cout << keypoints.size() << endl;
	//绘制
	drawKeypoints(img,	//原始图像
		keypoints,	//特征点向量
		img,	//输出图像
		Scalar(-1, -1, -1),	//特征点颜色  
		DrawMatchesFlags::DRAW_OVER_OUTIMG //绘制标记
		);
	// Display the corners
	cv::namedWindow("FAST检测的尖角");
	cv::imshow("FAST检测的尖角", img);
	return keypoints.size();
}

/************************************************************************/
/* 获取  vector<vector<Point>> contours 的区域像素（详细说明见我的csdn博客）：
http://blog.csdn.net/qq496830205/article/details/42673261
其中subRegions是返回参数。mask主要作为提取出来的图片的背景。*/
/************************************************************************/
void getContoursPixel(vector<vector<Point>> &contours, Mat srcImg,Vector<Mat> &subRegions, Mat mask)
{
	//vector<Mat> subRegions;
	for (int i = 0; i < contours.size(); i++)
	{
		// Get bounding box for contour
		Rect roi = boundingRect(contours[i]); // This is a OpenCV function
		// Create a mask for each contour to mask out that region from image.
		Mat maskTemp;// = Mat::zeros(srcImg.size(), CV_8UC1);
		//srcImg.copyTo(maskTemp);
		mask.copyTo(maskTemp);
		drawContours(maskTemp, contours, i, Scalar(255), CV_FILLED); // This is a OpenCV function
		// Extract region using mask for region
		Mat contourRegion;
		Mat imageROI;
		srcImg.copyTo(imageROI, maskTemp); // 'image' is the image you used to compute the contours.
		contourRegion = imageROI(roi);
		subRegions.push_back(contourRegion);
	}
//	cv::imshow("sub001", subRegions[contours.size()-1]);  //最大区域
// 	for (int i = 0; i < subregions.size(); i++)
// 	{
// 		cv::imshow("sub1" + i, subregions[i]);
// 	}	
}

/************************************************************************/
/* --输入参数thresholdedImg为分割好的图像（原图基础上分割的）。
-----获取火灾区域的特征：1、周长fireLength；2、面积fireArea；3、圆形度fireCircularity；
4、尖角个数fireConers；5、变化率（需要两张连续图片帧）fireRate；6、火灾区域的个数countOfFireContours
7、获取纹理特征: 求能量、熵、惯性矩、相关以及局部平稳性的均值和标准差作为最终10维纹理特征
*/
/************************************************************************/
void getFireFeatures(Mat srcImg, vector<vector<Point>> &contours, Mat &thresholdedImg)
{
	float fireLength, fireArea, fireCircularity, fireRate, countOfFireContours;
	int fireConers; //尖角个数fireConers
	/******************************** 显示所有的火灾区域及其特征（周长和面积；最后需要屏蔽）****************************************/
	getAllContoursFeatures(contours);

	/******************************** 获取面积最大的区域（用来获取特征）****************************************/
	vector<Point> biggestContour = getMaxOfContours(contours); //获取最大区域边界点
	//获取最大区域块（原图像）
	Vector<Mat> subRegionsSrcImg;
	imshow("SRCbiggestContourSrcImg", srcImg);
	getContoursPixel(contours, srcImg, subRegionsSrcImg, srcImg);
	Mat biggestContourSrcImg = subRegionsSrcImg[contours.size() - 1];
	imshow("biggestContourSrcImg", biggestContourSrcImg);  

	//获取最大区域块（将其显示在白色或者黑色区域上面）
	Vector<Mat> subRegionsThresholdedImg;
	cv::Mat whiteImgFromSrcImg(srcImg.size(), CV_8U, cv::Scalar(255));
	Mat mask = Mat::zeros(srcImg.size(), CV_8UC1);
	getContoursPixel(contours, whiteImgFromSrcImg, subRegionsThresholdedImg, mask);
	Mat biggestContourOnWhiteBlack = subRegionsThresholdedImg[contours.size() - 1];
	imshow("biggestContourThresholdedImg", biggestContourOnWhiteBlack);  

	/********************************获取特征****************************************/
	fireLength = arcLength(biggestContour,true);//1、周长
	fireArea = contourArea(biggestContour);//2、面积
	fireCircularity = fireLength*fireLength / (4 * PI*fireArea); //3、圆形度
	fireConers = getCountOfFireConers(biggestContourOnWhiteBlack);//4、尖角个数（最大火灾区域）
	//***5、变化率。带补充！！！！！！！！！
	countOfFireContours = contours.size(); //6、火灾区域的个数countOfFireContours，一般都是1
	//7、获取纹理特征: 求能量、熵、惯性矩、相关以及局部平稳性的均值和标准差作为最终10维纹理特征
	double GLCMFeatures[10];
	calGLCMNormalization(thresholdedImg, GLCMFeatures);  //利用FAST方法计算，单独头文件
	//8、灰度直方图特征：亮度均值和方差，直方图，投影
	Mat biggestContourSrcImgHSV;
	cout << "biggestContourSrcImg通道：" << biggestContourSrcImg.channels() << endl;
	cvtColor(biggestContourSrcImg, biggestContourSrcImg, CV_GRAY2BGR); //先转换为3通道BGR
	//imshow("biggestContourSrcImg00", biggestContourSrcImg);  //显示
	cvtColor(biggestContourSrcImg, biggestContourSrcImgHSV, CV_BGR2HSV); //3通道BGR转HSV（3通道）
	imshow("biggestContourSrcImgHSV", biggestContourSrcImgHSV);  //显示HSV图
	Scalar meanOfBiggestContourThresholdedImg;// 获取三个通道的均值 放在数组中//= mean(biggestContourSrcImgHSV); //
	Scalar stdDevOfBiggestContourThresholdedImg; //获取三个通道的标准差 放在数组中
	meanStdDev(biggestContourSrcImgHSV, meanOfBiggestContourThresholdedImg, stdDevOfBiggestContourThresholdedImg);
	//cout << meanOfBiggestContourThresholdedImg[0] << endl; //Hue色调=0
	//cout << meanOfBiggestContourThresholdedImg[1] << endl; //Saturation饱和度=0
	cout << "biggestContourSrcImgHSV亮度均值：" << meanOfBiggestContourThresholdedImg[2] << endl;  //Luminance亮度均值（非零）
	cout << "biggestContourSrcImgHSV亮度标准差：" << stdDevOfBiggestContourThresholdedImg[2] << endl;  //Luminance亮度标准差（非零）

	/********************************将特征写入文件，最后再封装为函数****************************************/
	ofstream ofile;
	ofile.open("特征记录.txt", ios::app);
	ofile << "1、周长fireLength, 2、面积fireArea, 3、圆形度fireCircularity\t, 4、尖角个数,\
			  6、火灾区域个数,7、能量均值,8、熵均值,9、惯性矩均值,10、相关性均值,11、局部平稳性均值" << endl;
	ofile << fireLength << "," << fireArea << "," << fireCircularity << "," << fireConers << countOfFireContours << "," \
		<< GLCMFeatures[0] << "," << GLCMFeatures[2] << "," << GLCMFeatures[4] \
		<< "," << GLCMFeatures[6] << "," << GLCMFeatures[8] << endl;
	ofile.close();
}

/************************************************************************/
/* 得到火灾区域及其大小 Get the contours of the connected components                                                                    */
/************************************************************************/
void getContours(Mat srcImg, Mat thresholdedImg)
{
	Mat thresholdedImgGray = thresholdedImg;
	//cvtColor(thresholdedImg, thresholdedImgGray, CV_BGR2GRAY);  //颜色空间的转CV_BGR2GRAY
	vector<vector<Point>> contours;
	findContours(thresholdedImgGray,  //image将变为含有轮廓的形状
		contours, // a vector of contours 
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

	/******************获取特征***********************/
	getFireFeatures(srcImg, contours, thresholdedImg);

	// 在白色图像上画出火灾区域（用黑线）
	cv::Mat result(thresholdedImg.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(0), // in black
		2); // with a thickness of 2
	cv::namedWindow("火灾区域（标注图）");
	cv::imshow("火灾区域（标注图）", result);

	// 排除面积过小和过大的区域
	int cmin = 20;  // minimum contour length
	int cmax = 10000; // maximum contour length
	std::vector<std::vector<cv::Point>>::const_iterator itc = contours.begin();  //通过迭代器，遍历每一个区域
	while (itc != contours.end())
	{
		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}

	// 在原始图像上面标出火灾区域
	cv::drawContours(srcImg, contours,
		-1, // draw all contours
		cv::Scalar(0, 0, 255), // in white
		2); // with a thickness of 2
	cv::namedWindow("火灾区域（原始图像）");
	cv::imshow("火灾区域（原始图像）", srcImg);
}


/************************************************************************/
/*    转换为黑白二值图：红色转换成白色，其他颜色转换成黑色     */
/************************************************************************/
void BGRto2Value(cv::Mat &image)
{
	// get iterators
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();
	for (; it != itend; ++it) {
		// 红色转换成白色，其他颜色转换成黑色
		if (((int)(*it)[0] + (int)(*it)[1] + abs((int)(*it)[2] - 255)) <= 250)
		{
			(*it)[0] = 255; (*it)[1] = 255; (*it)[2] = 255;
		}
		else
		{
			(*it)[0] = 0; (*it)[1] = 0; (*it)[2] = 0;
		}
	}
}

int main()
{
	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\基本操作：黑白化\\p4.jpg");
	cvtColor(srcImg, srcImg, CV_RGB2GRAY);  //转化为灰度图像
	cout << srcImg.channels() << endl;

	Mat medianFilterImg, thresholdedImg;
	namedWindow("原始图像");
	imshow("原始图像", srcImg);
	
	//1-1.滤波（中值滤波）
	medianBlur(srcImg, medianFilterImg, 5);
	namedWindow("中值滤波的结果");
	imshow("中值滤波的结果", medianFilterImg);

	//腐蚀 ToDo

	//彩色图像二值化
// 	if (result1.channels() == 3) //彩色图像，将其二值化（红色变为亮白色）。当灰色图像为3通道时，不能使用这个函数。
// 	{
// 		BGRto2Value(result1);
// 	}

	//1-2.增强（直方图均衡化）
// 	Mat result2(EqualizeHistColorImage(&(IplImage)result1));
// 	namedWindow("图像增强（对比度）");
// 	imshow("图像增强（对比度）", result2);

	//2.图像分割 creating a binary image by thresholding at the valley
	cv::threshold(medianFilterImg, thresholdedImg, 200, 255, cv::THRESH_BINARY);
	// Display the thresholded image
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image", thresholdedImg);
	cv::imwrite("binary.bmp", thresholdedImg);
/**/
	//3.提取
	getContours(srcImg, thresholdedImg);
	
	cv::waitKey();
	return 0;
}

