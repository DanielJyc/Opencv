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
	/* �Ҷ�ֱ��ͼ���������Ⱦ�ֵ�ͷ��� */
	/************************************************************************/
	void getMeanStdDevOfHist(Mat &biggestContourSrcImg, Scalar &meanOfBiggestContourThresholdedImgOfLuminance, Scalar &stdDevOfBiggestContourThresholdedImg)
	{
		Mat biggestContourSrcImgHSV;
		cout << "biggestContourSrcImgͨ����" << biggestContourSrcImg.channels() << endl;
		cvtColor(biggestContourSrcImg, biggestContourSrcImg, CV_GRAY2BGR); //��ת��Ϊ3ͨ��BGR
		cvtColor(biggestContourSrcImg, biggestContourSrcImgHSV, CV_BGR2HSV); //3ͨ��BGRתHSV��3ͨ����
		imshow("biggestContourSrcImgHSV", biggestContourSrcImgHSV);  //��ʾHSVͼ
		meanStdDev(biggestContourSrcImgHSV, meanOfBiggestContourThresholdedImgOfLuminance, stdDevOfBiggestContourThresholdedImg);
		//cout << meanOfBiggestContourThresholdedImg[0] << endl; //Hueɫ��=0
		//cout << meanOfBiggestContourThresholdedImg[1] << endl; //Saturation���Ͷ�=0
		cout << "biggestContourSrcImgHSV���Ⱦ�ֵ��" << meanOfBiggestContourThresholdedImgOfLuminance[2] << endl;  //Luminance���Ⱦ�ֵ�����㣩
		cout << "biggestContourSrcImgHSV���ȱ�׼�" << stdDevOfBiggestContourThresholdedImg[2] << endl;  //Luminance���ȱ�׼����㣩
	}

public:
	Mat srcImg;

	imageFeatures(Mat srcImg) {
		this->srcImg = srcImg;
	}




	/************************************************************************/
	/* --�������thresholdedImgΪ�ָ�õ�ͼ��ԭͼ�����Ϸָ�ģ���(ɾ����)
	-----��ȡ���������������1���ܳ�fireLength��2�����fireArea��3��Բ�ζ�fireCircularity��
	4����Ǹ���fireConers��5���仯�ʣ���Ҫ��������ͼƬ֡��fireRate��6����������ĸ���countOfFireContours
	7����ȡ��������: ���������ء����Ծء�����Լ��ֲ�ƽ���Եľ�ֵ�ͱ�׼����Ϊ����10ά��������
	*/
	/************************************************************************/
	void getFireFeatures(vector<vector<Point>> &contours, Mat biggestContourSrcImg, Mat biggestContourOnWhiteBlack)
	{
		float fireLength, fireArea, fireCircularity, fireRate, countOfFireContours;
		int fireConers; //��Ǹ���fireConers
		vector<Point> biggestContour = contours[contours.size() - 1];  //��ȡ������Ļ�������
		//getAllContoursFeatures(contours); //��ʾ���еĻ����������������ܳ�������������Ҫ���Σ�

		/********************************��ȡ����****************************************/
		fireLength = arcLength(biggestContour, true);//1���ܳ�
		fireArea = contourArea(biggestContour);//2�����
		fireCircularity = fireLength*fireLength / (4 * PI*fireArea); //3��Բ�ζ�
		fireConers = getCountOfFireConers(biggestContourOnWhiteBlack);//4����Ǹ���������������
		//***5���仯�ʡ������䣡����������������
		countOfFireContours = contours.size(); //6����������ĸ���countOfFireContours��һ�㶼��1
		//7����ȡ��������: ���������ء����Ծء�����Լ��ֲ�ƽ���Եľ�ֵ�ͱ�׼����Ϊ����10ά��������
		double GLCMFeatures[10];
		calGLCMNormalization(biggestContourSrcImg, GLCMFeatures);  //����FAST�������㣬����ͷ�ļ�
		//8���Ҷ�ֱ��ͼ����1�����Ⱦ�ֵ�ͷ���
		Scalar meanOfBiggestContourThresholdedImgOfLuminance;// ��ȡ����ͨ���ľ�ֵ ����������//= mean(biggestContourSrcImgHSV); //
		Scalar stdDevOfBiggestContourThresholdedImg; //��ȡ����ͨ���ı�׼�� ����������
		getMeanStdDevOfHist(biggestContourSrcImg, meanOfBiggestContourThresholdedImgOfLuminance, stdDevOfBiggestContourThresholdedImg);
		//9���Ҷ�ֱ��ͼ����1��ֱ��ͼ��ͶӰ����Test����ȡ��

		
		/********************************������д���ļ�������ٷ�װΪ����****************************************/
		ofstream ofile;
		ofile.open("������¼.csv", ios::app);
		ofile << "1���ܳ�fireLength, 2�����fireArea, 3��Բ�ζ�fireCircularity, 4����Ǹ���,6�������������,\
				 7��������ֵ,8���ؾ�ֵ,9�����Ծؾ�ֵ,10������Ծ�ֵ,11���ֲ�ƽ���Ծ�ֵ, \
				 12��Luminance���Ⱦ�ֵ�����㣩, 13��Luminance���ȱ�׼����㣩 "<< endl;
		ofile << fireLength << "," << fireArea << "," << fireCircularity << "," << fireConers << "," << countOfFireContours << "," \
			<< GLCMFeatures[0] << "," << GLCMFeatures[2] << "," << GLCMFeatures[4] << "," << GLCMFeatures[6] << "," << GLCMFeatures[8] << ","\
			<< meanOfBiggestContourThresholdedImgOfLuminance[2] << "," << stdDevOfBiggestContourThresholdedImg[2] << endl;
		ofile.close();

	}



};


#endif
