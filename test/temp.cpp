#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <fstream>
using namespace std;
using namespace cv;

void writeFile()
{
	ofstream ofile;               //��������ļ�
	ofile.open("������¼.txt", ios::app);     //��Ϊ����ļ���,���ļ�ĩβд�룬�����ǡ���Ĭ��Ϊ���ǣ�
	ofile << "f1\tf2\tf3" << endl;   //д���ļ�
	ofile.close();                //�ر��ļ�
}

void findTheConners()
{
	Mat biggestContourOnWhite = imread("findConers2.jpg");
	std::vector<cv::KeyPoint> keypoints; //�����������
	keypoints.clear();
	FastFeatureDetector fast(200); //�����ֵ
	fast.detect(biggestContourOnWhite, keypoints); //���м��
	cout << keypoints.size() << endl;
	//����
	drawKeypoints(biggestContourOnWhite,	//ԭʼͼ��
		keypoints,	//����������
		biggestContourOnWhite,	//���ͼ��
		Scalar(255, 0, 0),	//��������ɫ
		DrawMatchesFlags::DRAW_OVER_OUTIMG //���Ʊ��
		);
	// Display the corners
	cv::namedWindow("FAST Features");
	cv::imshow("FAST Features", biggestContourOnWhite);
}
void getContours(Mat thresholdedImg)
{
	Mat image;
	cvtColor(thresholdedImg, image, CV_BGR2GRAY);  //��ɫ�ռ��תCV_BGR2GRAY

	vector<vector<Point>> contours;
	findContours(image,
		contours, // a vector of contours 
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

	vector<Mat> subregions;
	// contours_final is as given above in your code
	for (int i = 0; i < contours.size(); i++)
	{
		// Get bounding box for contour
		Rect roi = boundingRect(contours[i]); // This is a OpenCV function

		// Create a mask for each contour to mask out that region from image.
		Mat mask = Mat::zeros(image.size(), CV_8UC1);
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

int main(){
	// 	writeFile();
	//	findTheConners();
	Mat thresholded;
	Mat srcImg = imread("F:\\kuaipan\\Visual Studio\\Projects\\grabFire\\�����������ڰ׻�\\p4.jpg");

	cv::threshold(srcImg, thresholded, 200, 255, cv::THRESH_BINARY);
	// Display the thresholded image
	cv::namedWindow("Binary Image");
	cv::imshow("Binary Image", thresholded);
	cv::imwrite("binary.bmp", thresholded);
	//3.��ȡ
	getContours(thresholded);

	waitKey();
	return 0;
}