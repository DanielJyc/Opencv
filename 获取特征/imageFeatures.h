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
	float featuresHist[5]; //5��ֱ��ͼ������features��0-50��50-100��100-150��150-200��200-255�� ͨ������get5HistFeatures��ɡ�

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

	/************************************************************************/
	/* ��ȡ������ ����λ��centerOffset������ͼƬ�� ��sqrt((x - xPreFrame)*(x - xPreFrame) - (y - yPreFrame)*(y - yPreFrame))     */
	/************************************************************************/
	float getFeature_centerOffset(vector<vector<Point>> &contours, vector<vector<Point>> &contoursPreviousFrame)
	{
		Moments mom = moments(contours[contours.size() - 1]);
		Moments momPreFrame = moments(contoursPreviousFrame[contoursPreviousFrame.size() - 1]);
		cout << mom.m10 / mom.m00 << "-------" << mom.m10 / mom.m00 << endl;
		float x = mom.m10 / mom.m00, y = mom.m01 / mom.m00;
		float xPreFrame = momPreFrame.m10 / momPreFrame.m00, yPreFrame = momPreFrame.m01 / momPreFrame.m00;
		float centerOffset = sqrt((x - xPreFrame)*(x - xPreFrame) - (y - yPreFrame)*(y - yPreFrame));
		//For test ��ԭͼ���滭������
// 		circle(srcImg, Point(x, y), 2, Scalar(0), 2);
// 		imshow("�����ʵ�", srcImg);
// 		circle(srcImgPreviousFrameInitial, Point(xPreFrame, yPreFrame), 2, Scalar(0), 2);
// 		imshow("�����ʵ�srcImgPreviousFrameInitial", srcImgPreviousFrameInitial);
		return centerOffset;
	}

	/************************************************************************/
	/* ��ȡֱ��ͼͳ��������5��ֱ��ͼ������features��0-50��50-100��100-150��150-200��200-255   */
	/************************************************************************/
	void get5HistFeatures(Mat srcImg)
	{
		float range[] = { 0, 256 };
		const float * histRange = { range };
		Mat hist;
		int histSize = 256;
		calcHist(&srcImg, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
		//��ȡ5��ֱ��ͼ������features��0-50��50-100��100-150��150-200��200-255
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
		//��ȡ����
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
	/* --�������thresholdedImgΪ�ָ�õ�ͼ��ԭͼ�����Ϸָ�ģ���(ɾ����)
	-----��ȡ���������������1���ܳ�fireLength��2�����fireArea��3��Բ�ζ�fireCircularity��
	4����Ǹ���fireConers��5���仯�ʣ���Ҫ��������ͼƬ֡��fireRate��6����������ĸ���countOfFireContours
	7����ȡ��������: ���������ء����Ծء�����Լ��ֲ�ƽ���Եľ�ֵ�ͱ�׼����Ϊ����10ά��������
	*/
	/************************************************************************/
	void getFireFeatures(vector<vector<Point>> &contours, vector<vector<Point>> &contoursPreviousFrame, Mat biggestContourSrcImg, Mat biggestContourOnWhiteBlack)
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
		//5����ȡ����仯��areasChangeRateFeature������λ��centerOffset������ͼƬ��
		//����仯��(S-S_pre)/S_pre
		float areasChangeRateFeature = (contourArea(contours[contours.size() - 1]) - contourArea(contoursPreviousFrame[contoursPreviousFrame.size() - 1])) / contourArea(contoursPreviousFrame[contoursPreviousFrame.size() - 1]);
		float centerOffsetFeature = getFeature_centerOffset(contours, contoursPreviousFrame);
		countOfFireContours = contours.size(); //6����������ĸ���countOfFireContours��һ�㶼��1
		//7����ȡ��������: ���������ء����Ծء�����Լ��ֲ�ƽ���Եľ�ֵ�ͱ�׼����Ϊ����10ά��������
		double GLCMFeatures[10];
		calGLCMNormalization(biggestContourSrcImg, GLCMFeatures);  //����FAST�������㣬����ͷ�ļ�
		//8���Ҷ�ֱ��ͼ����1�����Ⱦ�ֵ�ͷ���
		Scalar meanOfBiggestContourThresholdedImgOfLuminance;// ��ȡ����ͨ���ľ�ֵ ����������//= mean(biggestContourSrcImgHSV); //
		Scalar stdDevOfBiggestContourThresholdedImg; //��ȡ����ͨ���ı�׼�� ����������
		getMeanStdDevOfHist(biggestContourSrcImg, meanOfBiggestContourThresholdedImgOfLuminance, stdDevOfBiggestContourThresholdedImg);
		//9����ȡֱ��ͼͳ��������5��ֱ��ͼ������features��0-50��50-100��100-150��150-200��200-255
		get5HistFeatures(srcImg); //����Ҷ�ͼ�� ,��������featuresHist������

		/********************************������д���ļ�������ٷ�װΪ����****************************************/
		ofstream ofile;
		ofile.open("������¼.csv", ios::app);
		ofile << "1���ܳ�fireLength, 2�����fireArea, 3��Բ�ζ�fireCircularity, 4����Ǹ���, \
				 5-1������仯��, 5-2������λ��, 6�������������,\
				 7��������ֵ, 8���ؾ�ֵ, 9�����Ծؾ�ֵ, 10������Ծ�ֵ, 11���ֲ�ƽ���Ծ�ֵ, \
				 12��Luminance���Ⱦ�ֵ�����㣩, 13��Luminance���ȱ�׼����㣩, \
				 14���Ҷ�0-50, 15���Ҷ�50-100, 16���Ҷ�100-150, 17���Ҷ�150-200, 18���Ҷ�200-255"<< endl;
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
