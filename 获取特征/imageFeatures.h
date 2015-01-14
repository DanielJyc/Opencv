#if !defined IMGFEATURES
#define IMGFEATURES

#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
#include "featuresGLCM.h"


using namespace std;
using namespace cv;

#define PI 3.1415926

class imageFeatures {

private:
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

public:
	Mat srcImg;

	imageFeatures(Mat srcImg) {
		this->srcImg = srcImg;
	}




	/************************************************************************/
	/* --输入参数thresholdedImg为分割好的图像（原图基础上分割的）。(删除了)
	-----获取火灾区域的特征：1、周长fireLength；2、面积fireArea；3、圆形度fireCircularity；
	4、尖角个数fireConers；5、变化率（需要两张连续图片帧）fireRate；6、火灾区域的个数countOfFireContours
	7、获取纹理特征: 求能量、熵、惯性矩、相关以及局部平稳性的均值和标准差作为最终10维纹理特征
	*/
	/************************************************************************/
	void getFireFeatures(vector<vector<Point>> &contours, Mat biggestContourSrcImg, Mat biggestContourOnWhiteBlack)
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
		//***5、变化率。带补充！！！！！！！！！
		countOfFireContours = contours.size(); //6、火灾区域的个数countOfFireContours，一般都是1
		//7、获取纹理特征: 求能量、熵、惯性矩、相关以及局部平稳性的均值和标准差作为最终10维纹理特征
		double GLCMFeatures[10];
		calGLCMNormalization(biggestContourSrcImg, GLCMFeatures);  //利用FAST方法计算，单独头文件
		//8、灰度直方图特征1：亮度均值和方差
		Scalar meanOfBiggestContourThresholdedImgOfLuminance;// 获取三个通道的均值 放在数组中//= mean(biggestContourSrcImgHSV); //
		Scalar stdDevOfBiggestContourThresholdedImg; //获取三个通道的标准差 放在数组中
		getMeanStdDevOfHist(biggestContourSrcImg, meanOfBiggestContourThresholdedImgOfLuminance, stdDevOfBiggestContourThresholdedImg);
		//9、灰度直方图特征1：直方图，投影（从Test中提取）

		
		/********************************将特征写入文件，最后再封装为函数****************************************/
		ofstream ofile;
		ofile.open("特征记录.csv", ios::app);
		ofile << "1、周长fireLength, 2、面积fireArea, 3、圆形度fireCircularity, 4、尖角个数,6、火灾区域个数,\
				 7、能量均值,8、熵均值,9、惯性矩均值,10、相关性均值,11、局部平稳性均值, \
				 12、Luminance亮度均值（非零）, 13、Luminance亮度标准差（非零） "<< endl;
		ofile << fireLength << "," << fireArea << "," << fireCircularity << "," << fireConers << "," << countOfFireContours << "," \
			<< GLCMFeatures[0] << "," << GLCMFeatures[2] << "," << GLCMFeatures[4] << "," << GLCMFeatures[6] << "," << GLCMFeatures[8] << ","\
			<< meanOfBiggestContourThresholdedImgOfLuminance[2] << "," << stdDevOfBiggestContourThresholdedImg[2] << endl;
		ofile.close();

	}



};


#endif
