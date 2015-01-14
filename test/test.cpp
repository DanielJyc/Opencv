#include<fstream>
#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include "imageComparator.h"
//#include<iomanip>
using namespace std;
using namespace cv;

void writeFile()
{
	ofstream ofile;               //定义输出文件
	ofile.open("特征记录.txt", ios::app);     //作为输出文件打开,从文件末尾写入，不覆盖。（默认为覆盖）
	ofile << "f1\tf2\tf3" << endl;   //写入文件
	ofile.close();                //关闭文件
}

void findTheConners()
{
	Mat biggestContourOnWhite = imread("findConers2.jpg");
	std::vector<cv::KeyPoint> keypoints; //特征点的向量
	keypoints.clear();
	FastFeatureDetector fast(200); //检测阈值
	fast.detect(biggestContourOnWhite, keypoints); //进行检测
	cout << keypoints.size() << endl;
	//绘制
	drawKeypoints(biggestContourOnWhite,	//原始图像
		keypoints,	//特征点向量
		biggestContourOnWhite,	//输出图像
		Scalar(255, 0, 0),	//特征点颜色
		DrawMatchesFlags::DRAW_OVER_OUTIMG //绘制标记
		);
	// Display the corners
	cv::namedWindow("FAST Features");
	cv::imshow("FAST Features", biggestContourOnWhite);
}

void getContoursPixel(vector<vector<Point>> &contours, Mat &thresholdedImg)
{
	vector<Mat> subregions;
	// contours_final is as given above in your code
	for (int i = 0; i < contours.size(); i++)
	{
		// Get bounding box for contour
		Rect roi = boundingRect(contours[i]); // This is a OpenCV function

		// Create a mask for each contour to mask out that region from image.
		Mat mask = Mat::zeros(thresholdedImg.size(), CV_8UC1);
		drawContours(mask, contours, i, Scalar(255), CV_FILLED); // This is a OpenCV function

		// At this point, mask has value of 255 for pixels within the contour and value of 0 for those not in contour.

		// Extract region using mask for region
		Mat contourRegion;
		Mat imageROI;
		thresholdedImg.copyTo(imageROI, mask); // 'image' is the image you used to compute the contours.
		contourRegion = imageROI(roi);
		// Mat maskROI = mask(roi); // Save this if you want a mask for pixels within the contour in contourRegion. 

		// Store contourRegion. contourRegion is a rectangular image the size of the bounding rect for the contour 
		// BUT only pixels within the contour is visible. All other pixels are set to (0,0,0).
		subregions.push_back(contourRegion);
	}
	for (int i = 0; i < subregions.size(); i++)
	{
		cv::imshow("sub1" + i, subregions[i]);
	}
}

void getContours(Mat thresholdedImg)
{
	Mat image;
	cvtColor(thresholdedImg, image, CV_BGR2GRAY);  //颜色空间的转CV_BGR2GRAY

	vector<vector<Point>> contours;
	findContours(image,
		contours, // a vector of contours 
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

	getContoursPixel(contours, thresholdedImg);

}

void getContoursPixTest()
{

	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\基本操作：黑白化\\p4.jpg");
	Mat result1, thresholded;
	namedWindow("原始图像");
	imshow("原始图像", srcImg);

	//1-1.滤波（中值滤波）
	medianBlur(srcImg, result1, 5);
	namedWindow("中值滤波的结果");
	imshow("中值滤波的结果", result1);

	//2.图像分割 creating a binary image by thresholding at the valley
	cv::threshold(result1, thresholded, 200, 255, cv::THRESH_BINARY);
	// Display the thresholded image
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image", thresholded);
	cv::imwrite("binary.bmp", thresholded);

	//3.提取
	getContours(thresholded);
}

void getMeanOfLight_Test(Mat &srcImg)
{
	cvtColor(srcImg, srcImg, CV_RGB2GRAY);
	cout << srcImg.channels() << endl;
	cvtColor(srcImg, srcImg, CV_GRAY2RGB);
	cvtColor(srcImg, srcImg, CV_BGR2HSV);
	cout << srcImg.channels() << endl;

	vector<Mat> channels;
	split(srcImg, channels);
	Scalar m0 = mean(channels[2]);
	printf("%f\n", m0[0]);

	Scalar m = mean(srcImg);
	cout << m[0] << endl;
	cout << m[1] << endl;
	cout << m[2] << endl;
	imshow("原始图像1", srcImg);
}

float * get5HistFeatures(Mat srcImg)
{
	float featuresHist[5] = { 0 };
	float range[] = { 0, 256 };
	const float * histRange = { range };
	Mat hist;
	int histSize = 256;
	calcHist(&srcImg, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
	//获取5个直方图特征：features：0-50，50-100，100-150，150-200，200-255
	for (int i = 0; i < 50; i++)
	{
		featuresHist[0] += hist.at<float>(i);
	}
	for (int i = 50; i < 100; i++)
	{
		featuresHist[1] += hist.at<float>(i);
	}
	for (int i = 100; i < 150; i++)
	{
		featuresHist[2] += hist.at<float>(i);
	}
	for (int i = 150; i < 200; i++)
	{
		featuresHist[3] += hist.at<float>(i);
	}
	for (int i = 200; i < 256; i++)
	{
		featuresHist[4] += hist.at<float>(i);
	}
	//求取概率
	for (int i = 0; i < 5; i++)
	{
		featuresHist[i] = featuresHist[i] / sum(hist)[0];
	}
	return featuresHist;
}

void getEveryPixel(Mat &srcImg)
{
	Mat_<int> srcImg2 = srcImg;
	int i, j;
	for (i = 0; i < srcImg.rows; i++)
	{
		for (j = 0; j < srcImg.cols; j++)
		{
			//Scalar scal = cvGet2D(srcImg, i, j);
			//printf("(%f,%f,%f) \n", scal.val[0], scal.val[1], scal.val[2]); //黑白一个通道，彩色3通道
			cout << srcImg2(i, j) << endl;
		}
	}
	//cout << srcImg2(23,4) << endl;
	//cout << srcImg.size() << endl;
}

int main(){
	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\基本操作：黑白化\\p4.jpg");
	imshow("原始图像0", srcImg);
	// 	writeFile(); //写入文件
//	findTheConners();
//	getContoursPixTest(); //
//	getMeanOfLight_Test(srcImg); //亮度均值
	//getEveryPixel(srcImg); //获取每个像素
	


	//1、获取直方图统计特征：5个直方图特征：features：0-50，50-100，100-150，150-200，200-255
	cvtColor(srcImg, srcImg, CV_BGR2GRAY);
	float * featuresHist = get5HistFeatures(srcImg);
	cout << featuresHist[0] << ";" << featuresHist[1] << ";" << featuresHist[2] << ";" << featuresHist[3] << ";" << featuresHist[4] << ";" << endl;

	//2、通过比较直方图检索相似图片(注：两幅图像均为RGB图像)
	//巴氏系数（Bhattacharyyacoefficient）算法:http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
	//值越小越相似(0, 1)，所以结果用1 相减得到这个结果
	cv::Mat image = cv::imread("../images/waves.jpg");
	ImageComparator c;
	c.setReferenceImage(image);

	// Test
	cv::Mat input = cv::imread("../images/waves2.jpg");
	cout << input.size() << ":" << input.channels() << endl;
	cout << "waves vs waves2（由waves截图得到）: " << c.compare(input) << endl;
	// Read an image and compare it with reference
	cout << "waves vs waves: " << c.compare(image) << endl;
	cvtColor(srcImg, srcImg, CV_GRAY2RGB); 
	cout << "waves vs src: " << c.compare(srcImg) << endl;

	//3、求两幅图像的 变化率fireRate 和 中心位移 

	waitKey();
	return 0;
}