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
	/*                 �ȽϺ���                                             */
	/************************************************************************/
	static bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
		double i = fabs(contourArea(cv::Mat(contour1)));
		double j = fabs(contourArea(cv::Mat(contour2)));
		return (i < j);
	}

	/************************************************************************/
	/* ��������������ԭʼͼ����ʾ�������� */
	/************************************************************************/
	void showFireAreaOnSrcImg(Mat &thresholdedImg, vector<vector<Point>> &contours, Mat srcImg)
	{
		/******************����Ϊ���Դ���***********************/
		// �ڰ�ɫͼ���ϻ������������ú��ߣ�
		cv::Mat result(thresholdedImg.size(), CV_8U, cv::Scalar(255));
		cv::drawContours(result, contours,
			-1, // draw all contours
			cv::Scalar(0), // in black
			2); // with a thickness of 2
		cv::namedWindow("�������򣨱�עͼ��");
		cv::imshow("�������򣨱�עͼ��", result);

		// �ų������С�͹��������
		int cmin = 10;  // minimum contour length
		int cmax = 10000; // maximum contour length
		std::vector<std::vector<cv::Point>>::const_iterator itc = contours.begin();  //ͨ��������������ÿһ������
		while (itc != contours.end())
		{
			if (itc->size() < cmin || itc->size() > cmax)
				itc = contours.erase(itc);
			else
				++itc;
		}
		// ��ԭʼͼ����������������
		Mat	srcImgWithFireMarked;
		srcImg.copyTo(srcImgWithFireMarked);
		cv::drawContours(srcImgWithFireMarked, contours,
			-1, // draw all contours
			cv::Scalar(0, 0, 255), // in white
			2); // with a thickness of 2
		cv::namedWindow("��������ԭʼͼ��");
		cv::imshow("��������ԭʼͼ��", srcImgWithFireMarked);
	}

public:
	Mat srcImg;

	imageContours(Mat srcImg) {
		this->srcImg = srcImg;
	}

	/************************************************************************/
	/* �õ������������С(��С�����˳��)                             */
	/************************************************************************/
	vector<vector<Point>> getContours(Mat srcImg, Mat thresholdedImg)
	{
		Mat thresholdedImgGray = thresholdedImg;
		//cvtColor(thresholdedImg, thresholdedImgGray, CV_BGR2GRAY);  //��ɫ�ռ��תCV_BGR2GRAY
		vector<vector<Point>> contours;
		findContours(thresholdedImgGray,  //image����Ϊ������������״
			contours, // a vector of contours 
			CV_RETR_EXTERNAL, // retrieve the external contours
			CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours
		std::sort(contours.begin(), contours.end(), compareContourAreas);  //�������������.compareContourAreas Ϊ�ȽϺ���

		//��ʾÿ�������������ܳ�
		for (int i = 0; i < contours.size(); i++)
		{
			printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));
		}
		showFireAreaOnSrcImg(thresholdedImg, contours, srcImg);  //��ԭʼͼ����ʾ��������
		return contours;
	}


	/************************************************************************/
	/*  ��ȡ������Ļ���������������ź����contours  */
	/************************************************************************/
	vector<Point> getMaxOfContours(vector<vector<Point>> &contours)
	{
		std::vector<cv::Point> biggestContour = contours[contours.size() - 1]; // ��ȡ�������
		std::vector<cv::Point> smallestContour = contours[0]; // ��ȡ��С����
		cout << "�������" << contourArea(biggestContour) << endl;
		return biggestContour;
	}

	/************************************************************************/
	/* ��ȡ  vector<vector<Point>> contours ���������أ���ϸ˵�����ҵ�csdn���ͣ���
	http://blog.csdn.net/qq496830205/article/details/42673261
	����subRegions�Ƿ��ز�����mask��Ҫ��Ϊ��ȡ������ͼƬ�ı�����
	��backgroundImg������ȡ����һ��ΪsrcImg��*/
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
	//	cv::imshow("sub001", subRegions[contours.size()-1]);  //�������
	// 	for (int i = 0; i < subregions.size(); i++)
	// 	{
	// 		cv::imshow("sub1" + i, subregions[i]);
	// 	}	
	}

	/************************************************************************/
	/*   ��ȡ�������飨ԭͼ��                         */
	/************************************************************************/
	Mat getBiggestContourOnSrcImg(vector<vector<Point>> contours)
	{
		//vector<Point> biggestContour = getMaxOfContours(contours); //��ȡ�������߽��
		//��ȡ�������飨ԭͼ��
		Vector<Mat> subRegionsSrcImg;
		//imshow("SRCbiggestContourSrcImg", srcImg);
		getContoursPixel(contours, srcImg, subRegionsSrcImg, srcImg);
		Mat biggestContourSrcImg = subRegionsSrcImg[contours.size() - 1];
		return biggestContourSrcImg;
	}

	/************************************************************************/
	/*    ��ȡ�������飨������ʾ�ڰ�ɫ���ߺ�ɫ�������棩                */
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
