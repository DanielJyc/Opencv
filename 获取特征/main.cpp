#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
#include "imageContours.h"
#include "imageFeatures.h"


using namespace std;
using namespace cv;

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

/************************************************************************/
/* 初步处理                                                                     */
/************************************************************************/
void simpleProcess(Mat &srcImg, Mat &thresholdedImg)
{
	cvtColor(srcImg, srcImg, CV_RGB2GRAY);  //转化为灰度图像
	cout << srcImg.channels() << endl;
	imshow("原始图像", srcImg);

	//1-1.滤波（中值滤波）
	Mat medianFilterImg;
	medianBlur(srcImg, medianFilterImg, 5);
	namedWindow("中值滤波的结果");
	imshow("中值滤波的结果", medianFilterImg);

	//腐蚀 ToDo

	//2.图像分割 creating a binary image by thresholding at the valley
	cv::threshold(medianFilterImg, thresholdedImg, 200, 255, cv::THRESH_BINARY);  //阈值分割（后面再修改为区域分割法）
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image", thresholdedImg);
	cv::imwrite("binary.bmp", thresholdedImg);
}

int main()
{
	Mat thresholdedImg, thresholdedImgPreviousFrame;
	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\基本操作：黑白化\\p4.jpg");
	Mat srcImgPreviousFrame = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\基本操作：黑白化\\group.jpg");
	BGRto2Value(srcImgPreviousFrame);  //针对彩色图像
	cout << srcImgPreviousFrame.channels() << endl;
	imshow("srcImgPreviousFrame", srcImgPreviousFrame);

	//2.初步处理
	simpleProcess(srcImg, thresholdedImg);
	simpleProcess(srcImgPreviousFrame, thresholdedImgPreviousFrame);
	
	//3-1.提取区域srcImg
	imageContours imgContours(srcImg);
	vector<vector<Point>>  contours = imgContours.getContours(srcImg, thresholdedImg); //得到火灾区域及其大小(由小到大的顺序) 
	Mat biggestContourOnSrcImg = imgContours.getBiggestContourOnSrcImg(contours); //获取最大区域块（原图像）  
	Mat biggestContourOnWhiteBlack = imgContours.getBiggestContourOnWhiteBlack(contours);  //获取最大区域块（将其显示在白色或者黑色区域上面） 
	imshow("biggestContourOnSrcImg", biggestContourOnSrcImg);
	imshow("biggestContourOnWhiteBlack", biggestContourOnWhiteBlack);
	imshow("src", srcImg); //测试是否把原图像改变了。结果发现没有改变。——正常
	
	//3-2.提取区域srcImgPreviousFrame
	imageContours imgContoursPreviousFrame(srcImgPreviousFrame);
	vector<vector<Point>>  contoursPreviousFrame = imgContoursPreviousFrame.getContours(srcImgPreviousFrame, thresholdedImgPreviousFrame); //得到火灾区域及其大小(由小到大的顺序) 
	Mat biggestContourOnSrcImgPreviousFrame = imgContoursPreviousFrame.getBiggestContourOnSrcImg(contoursPreviousFrame); //获取最大区域块（原图像）  
	Mat biggestContourOnWhiteBlackPreviousFrame = imgContoursPreviousFrame.getBiggestContourOnWhiteBlack(contoursPreviousFrame);  //获取最大区域块（将其显示在白色或者黑色区域上面） 
	imshow("biggestContourOnSrcImgPreviousFrame", biggestContourOnSrcImgPreviousFrame);
	imshow("biggestContourOnWhiteBlackPreviousFrame", biggestContourOnWhiteBlackPreviousFrame);
	imshow("srcPreviousFrame", srcImgPreviousFrame);


	cout << "区域提取完成。" << endl;

	//4.提取特征 ：提取的特征将写入文件中
	imageFeatures imgFeatures(srcImg);
	imgFeatures.getFireFeatures(contours, biggestContourOnSrcImg, biggestContourOnWhiteBlack);

	cv::waitKey();
	return 0;
}





//彩色图像二值化
// 	if (result1.channels() == 3) //彩色图像，将其二值化（红色变为亮白色）。当灰色图像为3通道时，不能使用这个函数。
// 	{
// 		BGRto2Value(result1);
// 	}

//1-2.增强（直方图均衡化）
// 	Mat result2(EqualizeHistColorImage(&(IplImage)result1));
// 	namedWindow("图像增强（对比度）");
// 	imshow("图像增强（对比度）", result2);