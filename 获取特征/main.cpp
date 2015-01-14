#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  
#include "imageContours.h"
#include "imageFeatures.h"


using namespace std;
using namespace cv;

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

/************************************************************************/
/* ��������                                                                     */
/************************************************************************/
void simpleProcess(Mat &srcImg, Mat &thresholdedImg)
{
	cvtColor(srcImg, srcImg, CV_RGB2GRAY);  //ת��Ϊ�Ҷ�ͼ��
	cout << srcImg.channels() << endl;
	imshow("ԭʼͼ��", srcImg);

	//1-1.�˲�����ֵ�˲���
	Mat medianFilterImg;
	medianBlur(srcImg, medianFilterImg, 5);
	namedWindow("��ֵ�˲��Ľ��");
	imshow("��ֵ�˲��Ľ��", medianFilterImg);

	//��ʴ ToDo

	//2.ͼ��ָ� creating a binary image by thresholding at the valley
	cv::threshold(medianFilterImg, thresholdedImg, 200, 255, cv::THRESH_BINARY);  //��ֵ�ָ�������޸�Ϊ����ָ��
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image", thresholdedImg);
	cv::imwrite("binary.bmp", thresholdedImg);
}

int main()
{
	Mat thresholdedImg, thresholdedImgPreviousFrame;
	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\�����������ڰ׻�\\p4.jpg");
	Mat srcImgPreviousFrame = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\�����������ڰ׻�\\group.jpg");
	BGRto2Value(srcImgPreviousFrame);  //��Բ�ɫͼ��
	cout << srcImgPreviousFrame.channels() << endl;
	imshow("srcImgPreviousFrame", srcImgPreviousFrame);

	//2.��������
	simpleProcess(srcImg, thresholdedImg);
	simpleProcess(srcImgPreviousFrame, thresholdedImgPreviousFrame);
	
	//3-1.��ȡ����srcImg
	imageContours imgContours(srcImg);
	vector<vector<Point>>  contours = imgContours.getContours(srcImg, thresholdedImg); //�õ������������С(��С�����˳��) 
	Mat biggestContourOnSrcImg = imgContours.getBiggestContourOnSrcImg(contours); //��ȡ�������飨ԭͼ��  
	Mat biggestContourOnWhiteBlack = imgContours.getBiggestContourOnWhiteBlack(contours);  //��ȡ�������飨������ʾ�ڰ�ɫ���ߺ�ɫ�������棩 
	imshow("biggestContourOnSrcImg", biggestContourOnSrcImg);
	imshow("biggestContourOnWhiteBlack", biggestContourOnWhiteBlack);
	imshow("src", srcImg); //�����Ƿ��ԭͼ��ı��ˡ��������û�иı䡣��������
	
	//3-2.��ȡ����srcImgPreviousFrame
	imageContours imgContoursPreviousFrame(srcImgPreviousFrame);
	vector<vector<Point>>  contoursPreviousFrame = imgContoursPreviousFrame.getContours(srcImgPreviousFrame, thresholdedImgPreviousFrame); //�õ������������С(��С�����˳��) 
	Mat biggestContourOnSrcImgPreviousFrame = imgContoursPreviousFrame.getBiggestContourOnSrcImg(contoursPreviousFrame); //��ȡ�������飨ԭͼ��  
	Mat biggestContourOnWhiteBlackPreviousFrame = imgContoursPreviousFrame.getBiggestContourOnWhiteBlack(contoursPreviousFrame);  //��ȡ�������飨������ʾ�ڰ�ɫ���ߺ�ɫ�������棩 
	imshow("biggestContourOnSrcImgPreviousFrame", biggestContourOnSrcImgPreviousFrame);
	imshow("biggestContourOnWhiteBlackPreviousFrame", biggestContourOnWhiteBlackPreviousFrame);
	imshow("srcPreviousFrame", srcImgPreviousFrame);


	cout << "������ȡ��ɡ�" << endl;

	//4.��ȡ���� ����ȡ��������д���ļ���
	imageFeatures imgFeatures(srcImg);
	imgFeatures.getFireFeatures(contours, biggestContourOnSrcImg, biggestContourOnWhiteBlack);

	cv::waitKey();
	return 0;
}





//��ɫͼ���ֵ��
// 	if (result1.channels() == 3) //��ɫͼ�񣬽����ֵ������ɫ��Ϊ����ɫ��������ɫͼ��Ϊ3ͨ��ʱ������ʹ�����������
// 	{
// 		BGRto2Value(result1);
// 	}

//1-2.��ǿ��ֱ��ͼ���⻯��
// 	Mat result2(EqualizeHistColorImage(&(IplImage)result1));
// 	namedWindow("ͼ����ǿ���Աȶȣ�");
// 	imshow("ͼ����ǿ���Աȶȣ�", result2);