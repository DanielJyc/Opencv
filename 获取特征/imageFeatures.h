#if !defined IMGFEATURES
#define IMGFEATURES

#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
#include "featuresGLCM.h"
//#include "imageComparator.h"


using namespace std;
using namespace cv;

#define PI 3.1415926

class imageFeatures {

private:
	float featuresHist[5]; //5个直方图特征：features：0-50，50-100，100-150，150-200，200-255； 通过函数get5HistFeatures完成。

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
	/* 灰度直方图特征：亮度均值和方差 */
	/************************************************************************/
	void getMeanStdDevOfHist(Mat &biggestContourSrcImg, Scalar &meanOfBiggestContourThresholdedImgOfLuminance, Scalar &stdDevOfBiggestContourThresholdedImg)
	{
		Mat biggestContourSrcImgHSV;
		cout << "biggestContourSrcImg通道：" << biggestContourSrcImg.channels() << endl;
		cvtColor(biggestContourSrcImg, biggestContourSrcImg, CV_GRAY2BGR); //先转换为3通道BGR
		cvtColor(biggestContourSrcImg, biggestContourSrcImgHSV, CV_BGR2HSV); //3通道BGR转HSV（3通道）
		imshow("biggestContourSrcImgHSV", biggestContourSrcImgHSV);  //显示HSV图
		meanStdDev(biggestContourSrcImgHSV, meanOfBiggestContourThresholdedImgOfLuminance, stdDevOfBiggestContourThresholdedImg);
		//cout << meanOfBiggestContourThresholdedImg[0] << endl; //Hue色调=0
		//cout << meanOfBiggestContourThresholdedImg[1] << endl; //Saturation饱和度=0
		cout << "biggestContourSrcImgHSV亮度均值：" << meanOfBiggestContourThresholdedImgOfLuminance[2] << endl;  //Luminance亮度均值（非零）
		cout << "biggestContourSrcImgHSV亮度标准差：" << stdDevOfBiggestContourThresholdedImg[2] << endl;  //Luminance亮度标准差（非零）
	}

	/************************************************************************/
	/* 获取特征： 质心位移centerOffset（两张图片） ：sqrt((x - xPreFrame)*(x - xPreFrame) - (y - yPreFrame)*(y - yPreFrame))     */
	/************************************************************************/
	float getFeature_centerOffset(vector<vector<Point>> &contours, vector<vector<Point>> &contoursPreviousFrame)
	{
		Moments mom = moments(contours[contours.size() - 1]);
		Moments momPreFrame = moments(contoursPreviousFrame[contoursPreviousFrame.size() - 1]);
		cout << mom.m10 / mom.m00 << "-------" << mom.m10 / mom.m00 << endl;
		float x = mom.m10 / mom.m00, y = mom.m01 / mom.m00;
		float xPreFrame = momPreFrame.m10 / momPreFrame.m00, yPreFrame = momPreFrame.m01 / momPreFrame.m00;
		float centerOffset = sqrt((x - xPreFrame)*(x - xPreFrame) - (y - yPreFrame)*(y - yPreFrame));
		//For test 在原图上面画出质心
// 		circle(srcImg, Point(x, y), 2, Scalar(0), 2);
// 		imshow("画出质点", srcImg);
// 		circle(srcImgPreviousFrameInitial, Point(xPreFrame, yPreFrame), 2, Scalar(0), 2);
// 		imshow("画出质点srcImgPreviousFrameInitial", srcImgPreviousFrameInitial);
		return centerOffset;
	}

	/************************************************************************/
	/* 获取直方图统计特征：5个直方图特征：features：0-50，50-100，100-150，150-200，200-255   */
	/************************************************************************/
	void get5HistFeatures(Mat srcImg)
	{
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
		//cout << featuresHist[0] << ";" << featuresHist[1] << ";" << featuresHist[2] << ";" << featuresHist[3] << ";" << featuresHist[4] << ";" << endl;
		//return featuresHist;
	}


public:
	Mat srcImg;

	imageFeatures(Mat srcImg) {
		this->srcImg = srcImg;
		featuresHist[5] = { 0 };
	}

	/************************************************************************/
	/* --输入参数thresholdedImg为分割好的图像（原图基础上分割的）。(删除了)
	-----获取火灾区域的特征：1、周长fireLength；2、面积fireArea；3、圆形度fireCircularity；
	4、尖角个数fireConers；5、变化率（需要两张连续图片帧）fireRate；6、火灾区域的个数countOfFireContours
	7、获取纹理特征: 求能量、熵、惯性矩、相关以及局部平稳性的均值和标准差作为最终10维纹理特征
	*/
	/************************************************************************/
	void getFireFeatures(vector<vector<Point>> &contours, vector<vector<Point>> &contoursPreviousFrame, Mat biggestContourSrcImg, Mat biggestContourOnWhiteBlack)
	{
		float fireLength, fireArea, fireCircularity, fireRate, countOfFireContours;
		int fireConers; //尖角个数fireConers
		vector<Point> biggestContour = contours[contours.size() - 1];  //获取面积最大的火灾区域。
		//getAllContoursFeatures(contours); //显示所有的火灾区域及其特征（周长和面积；最后需要屏蔽）

		/********************************获取特征****************************************/
		fireLength = arcLength(biggestContour, true);//1、周长
		fireArea = contourArea(biggestContour);//2、面积
		fireCircularity = fireLength*fireLength / (4 * PI*fireArea); //3、圆形度
		fireConers = getCountOfFireConers(biggestContourOnWhiteBlack);//4、尖角个数（最大火灾区域）
		//5、求取面积变化率areasChangeRateFeature和质心位移centerOffset（两张图片）
		//面积变化率(S-S_pre)/S_pre
		float areasChangeRateFeature = (contourArea(contours[contours.size() - 1]) - contourArea(contoursPreviousFrame[contoursPreviousFrame.size() - 1])) / contourArea(contoursPreviousFrame[contoursPreviousFrame.size() - 1]);
		float centerOffsetFeature = getFeature_centerOffset(contours, contoursPreviousFrame);
		countOfFireContours = contours.size(); //6、火灾区域的个数countOfFireContours，一般都是1
		//7、获取纹理特征: 求能量、熵、惯性矩、相关以及局部平稳性的均值和标准差作为最终10维纹理特征
		double GLCMFeatures[10];
		calGLCMNormalization(biggestContourSrcImg, GLCMFeatures);  //利用FAST方法计算，单独头文件
		//8、灰度直方图特征1：亮度均值和方差
		Scalar meanOfBiggestContourThresholdedImgOfLuminance;// 获取三个通道的均值 放在数组中//= mean(biggestContourSrcImgHSV); //
		Scalar stdDevOfBiggestContourThresholdedImg; //获取三个通道的标准差 放在数组中
		getMeanStdDevOfHist(biggestContourSrcImg, meanOfBiggestContourThresholdedImgOfLuminance, stdDevOfBiggestContourThresholdedImg);
		//9、获取直方图统计特征：5个直方图特征：features：0-50，50-100，100-150，150-200，200-255
		get5HistFeatures(srcImg); //输入灰度图像 ,特征放在featuresHist数组中

		/********************************将特征写入文件，最后再封装为函数****************************************/
		ofstream ofile;
		ofile.open("特征记录.csv", ios::app);
		ofile << "1、周长fireLength, 2、面积fireArea, 3、圆形度fireCircularity, 4、尖角个数, \
				 5-1、面积变化率, 5-2、质心位移, 6、火灾区域个数,\
				 7、能量均值, 8、熵均值, 9、惯性矩均值, 10、相关性均值, 11、局部平稳性均值, \
				 12、Luminance亮度均值（非零）, 13、Luminance亮度标准差（非零）, \
				 14、灰度0-50, 15、灰度50-100, 16、灰度100-150, 17、灰度150-200, 18、灰度200-255"<< endl;
		ofile << fireLength << "," << fireArea << "," << fireCircularity << "," << fireConers << "," \
			<< areasChangeRateFeature << "," << centerOffsetFeature<< "," << countOfFireContours << "," \
			<< GLCMFeatures[0] << "," << GLCMFeatures[2] << "," << GLCMFeatures[4] << "," << GLCMFeatures[6] << "," << GLCMFeatures[8] << ","\
			<< meanOfBiggestContourThresholdedImgOfLuminance[2] << "," << stdDevOfBiggestContourThresholdedImg[2] \
			<< "," << featuresHist[0] << "," << featuresHist[1] << "," << featuresHist[2] << "," << featuresHist[3] << "," << featuresHist[4] << endl;
		ofile.close();
		cout << featuresHist[0] << ";" << featuresHist[1] << ";" << featuresHist[2] << ";" << featuresHist[3] << ";" << featuresHist[4] << ";" << endl;

	}

};


#endif
