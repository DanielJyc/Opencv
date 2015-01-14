#if !defined IMGCONTOURS
#define IMGCONTOURS

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  

using namespace std;
using namespace cv;

class imageContours {

private:
	/************************************************************************/
	/*                 比较函数                                             */
	/************************************************************************/
	static bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
		double i = fabs(contourArea(cv::Mat(contour1)));
		double j = fabs(contourArea(cv::Mat(contour2)));
		return (i < j);
	}

	/************************************************************************/
	/* 画出火灾区域；在原始图像显示火灾区域 */
	/************************************************************************/
	void showFireAreaOnSrcImg(Mat &thresholdedImg, vector<vector<Point>> &contours, Mat srcImg)
	{
		/******************下面为测试代码***********************/
		// 在白色图像上画出火灾区域（用黑线）
		cv::Mat result(thresholdedImg.size(), CV_8U, cv::Scalar(255));
		cv::drawContours(result, contours,
			-1, // draw all contours
			cv::Scalar(0), // in black
			2); // with a thickness of 2
		cv::namedWindow("火灾区域（标注图）");
		cv::imshow("火灾区域（标注图）", result);

		// 排除面积过小和过大的区域
		int cmin = 10;  // minimum contour length
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
		Mat	srcImgWithFireMarked;
		srcImg.copyTo(srcImgWithFireMarked);
		cv::drawContours(srcImgWithFireMarked, contours,
			-1, // draw all contours
			cv::Scalar(0, 0, 255), // in white
			2); // with a thickness of 2
		cv::namedWindow("火灾区域（原始图像）");
		cv::imshow("火灾区域（原始图像）", srcImgWithFireMarked);
	}

public:
	Mat srcImg;

	imageContours(Mat srcImg) {
		this->srcImg = srcImg;
	}

	/************************************************************************/
	/* 得到火灾区域及其大小(由小到大的顺序)                             */
	/************************************************************************/
	vector<vector<Point>> getContours(Mat srcImg, Mat thresholdedImg)
	{
		Mat thresholdedImgGray = thresholdedImg;
		//cvtColor(thresholdedImg, thresholdedImgGray, CV_BGR2GRAY);  //颜色空间的转CV_BGR2GRAY
		vector<vector<Point>> contours;
		findContours(thresholdedImgGray,  //image将变为含有轮廓的形状
			contours, // a vector of contours 
			CV_RETR_EXTERNAL, // retrieve the external contours
			CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours
		std::sort(contours.begin(), contours.end(), compareContourAreas);  //对区域进行排序.compareContourAreas 为比较函数

		//显示每个区域的面积和周长
		for (int i = 0; i < contours.size(); i++)
		{
			printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));
		}
		showFireAreaOnSrcImg(thresholdedImg, contours, srcImg);  //在原始图像显示火灾区域
		return contours;
	}


	/************************************************************************/
	/*  获取面积最大的火灾区域。输入的是排好序的contours  */
	/************************************************************************/
	vector<Point> getMaxOfContours(vector<vector<Point>> &contours)
	{
		std::vector<cv::Point> biggestContour = contours[contours.size() - 1]; // 获取最大区域
		std::vector<cv::Point> smallestContour = contours[0]; // 获取最小区域
		cout << "最大区域" << contourArea(biggestContour) << endl;
		return biggestContour;
	}

	/************************************************************************/
	/* 获取  vector<vector<Point>> contours 的区域像素（详细说明见我的csdn博客）：
	http://blog.csdn.net/qq496830205/article/details/42673261
	其中subRegions是返回参数。mask主要作为提取出来的图片的背景。
	从backgroundImg上面提取区域（一般为srcImg）*/
	/************************************************************************/
	void getContoursPixel(vector<vector<Point>> &contours, Mat backgroundImg,Vector<Mat> &subRegions, Mat mask)
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
			backgroundImg.copyTo(imageROI, maskTemp); // 'image' is the image you used to compute the contours.
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
	/*   获取最大区域块（原图像）                         */
	/************************************************************************/
	Mat getBiggestContourOnSrcImg(vector<vector<Point>> contours)
	{
		//vector<Point> biggestContour = getMaxOfContours(contours); //获取最大区域边界点
		//获取最大区域块（原图像）
		Vector<Mat> subRegionsSrcImg;
		//imshow("SRCbiggestContourSrcImg", srcImg);
		getContoursPixel(contours, srcImg, subRegionsSrcImg, srcImg);
		Mat biggestContourSrcImg = subRegionsSrcImg[contours.size() - 1];
		return biggestContourSrcImg;
	}

	/************************************************************************/
	/*    获取最大区域块（将其显示在白色或者黑色区域上面）                */
	/************************************************************************/
	Mat getBiggestContourOnWhiteBlack(vector<vector<Point>> contours)
	{
		Vector<Mat> subRegionsThresholdedImg;
		cv::Mat whiteImgFromSrcImg(srcImg.size(), CV_8U, cv::Scalar(255));
		Mat mask = Mat::zeros(srcImg.size(), CV_8UC1);
		getContoursPixel(contours, whiteImgFromSrcImg, subRegionsThresholdedImg, mask);
		Mat biggestContourOnWhiteBlack = subRegionsThresholdedImg[contours.size() - 1];
		return biggestContourOnWhiteBlack;
	}
};


#endif
