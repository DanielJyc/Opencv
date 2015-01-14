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
/*                 �ȽϺ���                                             */
/************************************************************************/
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}

/************************************************************************/
/*  ��ȡ�����������                                                                    */
/************************************************************************/
vector<Point> getMaxOfContours(vector<vector<Point>> &contours)
{
	std::sort(contours.begin(), contours.end(), compareContourAreas);  //�������������.compareContourAreas Ϊ�ȽϺ���
	std::vector<cv::Point> biggestContour = contours[contours.size() - 1]; // ��ȡ�������
	std::vector<cv::Point> smallestContour = contours[0]; // ��ȡ��С����
	cout << "�������" << contourArea(biggestContour) << endl;
	return biggestContour;
}

/************************************************************************/
/* ��ʾ���еĻ����������������ܳ��������                             */
/************************************************************************/
void getAllContoursFeatures(vector<vector<Point>> &contours)
{
	// ��ʾÿ����ͨ����Ĵ�С
	std::cout << "Contours: " << contours.size() << std::endl;
	std::vector<vector<Point>>::const_iterator itContours = contours.begin();
	for (; itContours != contours.end(); ++itContours)
	{
		std::cout << "Size: " << itContours->size() << std::endl;
	}
	//��ʾÿ�������������ܳ�
	for (int i = 0; i < contours.size(); i++)
	{
		printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));
	}
}

/************************************************************************/
/* ��ȡͼƬimg�ļ�Ǹ���                                               */
/************************************************************************/
int getCountOfFireConers(Mat img)
{
	//�ҳ��ǵ��������ע��ͼ��
	std::vector<cv::KeyPoint> keypoints; //�����������
	keypoints.clear();
	FastFeatureDetector fast(250); //�����ֵ 200 40
	fast.detect(img, keypoints); //���м��
	cout << keypoints.size() << endl;
	//����
	drawKeypoints(img,	//ԭʼͼ��
		keypoints,	//����������
		img,	//���ͼ��
		Scalar(-1, -1, -1),	//��������ɫ  
		DrawMatchesFlags::DRAW_OVER_OUTIMG //���Ʊ��
		);
	// Display the corners
	cv::namedWindow("FAST���ļ��");
	cv::imshow("FAST���ļ��", img);
	return keypoints.size();
}

/************************************************************************/
/* ��ȡ  vector<vector<Point>> contours ���������أ���ϸ˵�����ҵ�csdn���ͣ���
http://blog.csdn.net/qq496830205/article/details/42673261
����subRegions�Ƿ��ز�����mask��Ҫ��Ϊ��ȡ������ͼƬ�ı�����*/
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
//	cv::imshow("sub001", subRegions[contours.size()-1]);  //�������
// 	for (int i = 0; i < subregions.size(); i++)
// 	{
// 		cv::imshow("sub1" + i, subregions[i]);
// 	}	
}

/************************************************************************/
/* --�������thresholdedImgΪ�ָ�õ�ͼ��ԭͼ�����Ϸָ�ģ���
-----��ȡ���������������1���ܳ�fireLength��2�����fireArea��3��Բ�ζ�fireCircularity��
4����Ǹ���fireConers��5���仯�ʣ���Ҫ��������ͼƬ֡��fireRate��6����������ĸ���countOfFireContours
7����ȡ��������: ���������ء����Ծء�����Լ��ֲ�ƽ���Եľ�ֵ�ͱ�׼����Ϊ����10ά��������
*/
/************************************************************************/
void getFireFeatures(Mat srcImg, vector<vector<Point>> &contours, Mat &thresholdedImg)
{
	float fireLength, fireArea, fireCircularity, fireRate, countOfFireContours;
	int fireConers; //��Ǹ���fireConers
	/******************************** ��ʾ���еĻ����������������ܳ�������������Ҫ���Σ�****************************************/
	getAllContoursFeatures(contours);

	/******************************** ��ȡ�����������������ȡ������****************************************/
	vector<Point> biggestContour = getMaxOfContours(contours); //��ȡ�������߽��
	//��ȡ�������飨ԭͼ��
	Vector<Mat> subRegionsSrcImg;
	imshow("SRCbiggestContourSrcImg", srcImg);
	getContoursPixel(contours, srcImg, subRegionsSrcImg, srcImg);
	Mat biggestContourSrcImg = subRegionsSrcImg[contours.size() - 1];
	imshow("biggestContourSrcImg", biggestContourSrcImg);  

	//��ȡ�������飨������ʾ�ڰ�ɫ���ߺ�ɫ�������棩
	Vector<Mat> subRegionsThresholdedImg;
	cv::Mat whiteImgFromSrcImg(srcImg.size(), CV_8U, cv::Scalar(255));
	Mat mask = Mat::zeros(srcImg.size(), CV_8UC1);
	getContoursPixel(contours, whiteImgFromSrcImg, subRegionsThresholdedImg, mask);
	Mat biggestContourOnWhiteBlack = subRegionsThresholdedImg[contours.size() - 1];
	imshow("biggestContourThresholdedImg", biggestContourOnWhiteBlack);  

	/********************************��ȡ����****************************************/
	fireLength = arcLength(biggestContour,true);//1���ܳ�
	fireArea = contourArea(biggestContour);//2�����
	fireCircularity = fireLength*fireLength / (4 * PI*fireArea); //3��Բ�ζ�
	fireConers = getCountOfFireConers(biggestContourOnWhiteBlack);//4����Ǹ���������������
	//***5���仯�ʡ������䣡����������������
	countOfFireContours = contours.size(); //6����������ĸ���countOfFireContours��һ�㶼��1
	//7����ȡ��������: ���������ء����Ծء�����Լ��ֲ�ƽ���Եľ�ֵ�ͱ�׼����Ϊ����10ά��������
	double GLCMFeatures[10];
	calGLCMNormalization(thresholdedImg, GLCMFeatures);  //����FAST�������㣬����ͷ�ļ�
	//8���Ҷ�ֱ��ͼ���������Ⱦ�ֵ�ͷ��ֱ��ͼ��ͶӰ
	Mat biggestContourSrcImgHSV;
	cout << "biggestContourSrcImgͨ����" << biggestContourSrcImg.channels() << endl;
	cvtColor(biggestContourSrcImg, biggestContourSrcImg, CV_GRAY2BGR); //��ת��Ϊ3ͨ��BGR
	//imshow("biggestContourSrcImg00", biggestContourSrcImg);  //��ʾ
	cvtColor(biggestContourSrcImg, biggestContourSrcImgHSV, CV_BGR2HSV); //3ͨ��BGRתHSV��3ͨ����
	imshow("biggestContourSrcImgHSV", biggestContourSrcImgHSV);  //��ʾHSVͼ
	Scalar meanOfBiggestContourThresholdedImg;// ��ȡ����ͨ���ľ�ֵ ����������//= mean(biggestContourSrcImgHSV); //
	Scalar stdDevOfBiggestContourThresholdedImg; //��ȡ����ͨ���ı�׼�� ����������
	meanStdDev(biggestContourSrcImgHSV, meanOfBiggestContourThresholdedImg, stdDevOfBiggestContourThresholdedImg);
	//cout << meanOfBiggestContourThresholdedImg[0] << endl; //Hueɫ��=0
	//cout << meanOfBiggestContourThresholdedImg[1] << endl; //Saturation���Ͷ�=0
	cout << "biggestContourSrcImgHSV���Ⱦ�ֵ��" << meanOfBiggestContourThresholdedImg[2] << endl;  //Luminance���Ⱦ�ֵ�����㣩
	cout << "biggestContourSrcImgHSV���ȱ�׼�" << stdDevOfBiggestContourThresholdedImg[2] << endl;  //Luminance���ȱ�׼����㣩

	/********************************������д���ļ�������ٷ�װΪ����****************************************/
	ofstream ofile;
	ofile.open("������¼.txt", ios::app);
	ofile << "1���ܳ�fireLength, 2�����fireArea, 3��Բ�ζ�fireCircularity\t, 4����Ǹ���,\
			  6�������������,7��������ֵ,8���ؾ�ֵ,9�����Ծؾ�ֵ,10������Ծ�ֵ,11���ֲ�ƽ���Ծ�ֵ" << endl;
	ofile << fireLength << "," << fireArea << "," << fireCircularity << "," << fireConers << countOfFireContours << "," \
		<< GLCMFeatures[0] << "," << GLCMFeatures[2] << "," << GLCMFeatures[4] \
		<< "," << GLCMFeatures[6] << "," << GLCMFeatures[8] << endl;
	ofile.close();
}

/************************************************************************/
/* �õ������������С Get the contours of the connected components                                                                    */
/************************************************************************/
void getContours(Mat srcImg, Mat thresholdedImg)
{
	Mat thresholdedImgGray = thresholdedImg;
	//cvtColor(thresholdedImg, thresholdedImgGray, CV_BGR2GRAY);  //��ɫ�ռ��תCV_BGR2GRAY
	vector<vector<Point>> contours;
	findContours(thresholdedImgGray,  //image����Ϊ������������״
		contours, // a vector of contours 
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

	/******************��ȡ����***********************/
	getFireFeatures(srcImg, contours, thresholdedImg);

	// �ڰ�ɫͼ���ϻ������������ú��ߣ�
	cv::Mat result(thresholdedImg.size(), CV_8U, cv::Scalar(255));
	cv::drawContours(result, contours,
		-1, // draw all contours
		cv::Scalar(0), // in black
		2); // with a thickness of 2
	cv::namedWindow("�������򣨱�עͼ��");
	cv::imshow("�������򣨱�עͼ��", result);

	// �ų������С�͹��������
	int cmin = 20;  // minimum contour length
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
	cv::drawContours(srcImg, contours,
		-1, // draw all contours
		cv::Scalar(0, 0, 255), // in white
		2); // with a thickness of 2
	cv::namedWindow("��������ԭʼͼ��");
	cv::imshow("��������ԭʼͼ��", srcImg);
}


/************************************************************************/
/*    ת��Ϊ�ڰ׶�ֵͼ����ɫת���ɰ�ɫ��������ɫת���ɺ�ɫ     */
/************************************************************************/
void BGRto2Value(cv::Mat &image)
{
	// get iterators
	cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();
	for (; it != itend; ++it) {
		// ��ɫת���ɰ�ɫ��������ɫת���ɺ�ɫ
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
	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\�����������ڰ׻�\\p4.jpg");
	cvtColor(srcImg, srcImg, CV_RGB2GRAY);  //ת��Ϊ�Ҷ�ͼ��
	cout << srcImg.channels() << endl;

	Mat medianFilterImg, thresholdedImg;
	namedWindow("ԭʼͼ��");
	imshow("ԭʼͼ��", srcImg);
	
	//1-1.�˲�����ֵ�˲���
	medianBlur(srcImg, medianFilterImg, 5);
	namedWindow("��ֵ�˲��Ľ��");
	imshow("��ֵ�˲��Ľ��", medianFilterImg);

	//��ʴ ToDo

	//��ɫͼ���ֵ��
// 	if (result1.channels() == 3) //��ɫͼ�񣬽����ֵ������ɫ��Ϊ����ɫ��������ɫͼ��Ϊ3ͨ��ʱ������ʹ�����������
// 	{
// 		BGRto2Value(result1);
// 	}

	//1-2.��ǿ��ֱ��ͼ���⻯��
// 	Mat result2(EqualizeHistColorImage(&(IplImage)result1));
// 	namedWindow("ͼ����ǿ���Աȶȣ�");
// 	imshow("ͼ����ǿ���Աȶȣ�", result2);

	//2.ͼ��ָ� creating a binary image by thresholding at the valley
	cv::threshold(medianFilterImg, thresholdedImg, 200, 255, cv::THRESH_BINARY);
	// Display the thresholded image
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image", thresholdedImg);
	cv::imwrite("binary.bmp", thresholdedImg);
/**/
	//3.��ȡ
	getContours(srcImg, thresholdedImg);
	
	cv::waitKey();
	return 0;
}

