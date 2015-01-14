#if !defined COLHISTOGRAM
#define IMGCONTOURS

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>  

using namespace std;
using namespace cv;

class imageContours {

private:

	int histSize[3];
	float hranges[2];
	const float* ranges[3];
	int channels[3];

public:

	/************************************************************************/
	/* 得到火灾区域及其大小 Get the contours of the connected components                                                                    */
	/************************************************************************/
	void getContours(Mat srcImg, Mat thresholdedImg)
	{
		Mat thresholdedImgGray = thresholdedImg;
		//cvtColor(thresholdedImg, thresholdedImgGray, CV_BGR2GRAY);  //颜色空间的转CV_BGR2GRAY
		vector<vector<Point>> contours;
		findContours(thresholdedImgGray,  //image将变为含有轮廓的形状
			contours, // a vector of contours 
			CV_RETR_EXTERNAL, // retrieve the external contours
			CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

		/******************获取特征***********************/
		//getFireFeatures(srcImg, contours, thresholdedImg);

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
		cv::drawContours(srcImg, contours,
			-1, // draw all contours
			cv::Scalar(0, 0, 255), // in white
			2); // with a thickness of 2
		cv::namedWindow("火灾区域（原始图像）");
		cv::imshow("火灾区域（原始图像）", srcImg);
	}



	imageContours() {

		// Prepare arguments for a color histogram
		histSize[0] = histSize[1] = histSize[2] = 256;
		hranges[0] = 0.0;    // BRG range
		hranges[1] = 255.0;
		ranges[0] = hranges; // all channels have the same range 
		ranges[1] = hranges;
		ranges[2] = hranges;
		channels[0] = 0;		// the three channels 
		channels[1] = 1;
		channels[2] = 2;
	}

	// Computes the histogram.
	cv::MatND getHistogram(const cv::Mat &image) {

		cv::MatND hist;

		// BGR color histogram
		hranges[0] = 0.0;    // BRG range
		hranges[1] = 255.0;
		channels[0] = 0;		// the three channels 
		channels[1] = 1;
		channels[2] = 2;

		// Compute histogram
		cv::calcHist(&image,
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			3,			// it is a 3D histogram
			histSize,	// number of bins
			ranges		// pixel value range
			);

		return hist;
	}

	// Computes the histogram.
	cv::SparseMat getSparseHistogram(const cv::Mat &image) {

		cv::SparseMat hist(3, histSize, CV_32F);

		// BGR color histogram
		hranges[0] = 0.0;    // BRG range
		hranges[1] = 255.0;
		channels[0] = 0;		// the three channels 
		channels[1] = 1;
		channels[2] = 2;

		// Compute histogram
		cv::calcHist(&image,
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			3,			// it is a 3D histogram
			histSize,	// number of bins
			ranges		// pixel value range
			);

		return hist;
	}

	// Computes the 2D ab histogram.
	// BGR source image is converted to Lab
	cv::MatND getabHistogram(const cv::Mat &image) {

		cv::MatND hist;

		// Convert to Lab color space
		cv::Mat lab;
		cv::cvtColor(image, lab, CV_BGR2Lab);

		// Prepare arguments for a 2D color histogram
		hranges[0] = -128.0;
		hranges[1] = 127.0;
		channels[0] = 1; // the two channels used are ab 
		channels[1] = 2;

		// Compute histogram
		cv::calcHist(&lab,
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			2,			// it is a 2D histogram
			histSize,	// number of bins
			ranges		// pixel value range
			);

		return hist;
	}

	// Computes the 1D Hue histogram with a mask.
	// BGR source image is converted to HSV
	cv::MatND getHueHistogram(const cv::Mat &image) {

		cv::MatND hist;

		// Convert to Lab color space
		cv::Mat hue;
		cv::cvtColor(image, hue, CV_BGR2HSV);

		// Prepare arguments for a 1D hue histogram
		hranges[0] = 0.0;
		hranges[1] = 180.0;
		channels[0] = 0; // the hue channel 

		// Compute histogram
		cv::calcHist(&hue,
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			1,			// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
			);

		return hist;
	}

	cv::Mat colorReduce(const cv::Mat &image, int div = 64) {

		int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
		// mask used to round the pixel value
		uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

		cv::Mat_<cv::Vec3b>::const_iterator it = image.begin<cv::Vec3b>();
		cv::Mat_<cv::Vec3b>::const_iterator itend = image.end<cv::Vec3b>();

		// Set output image (always 1-channel)
		cv::Mat result(image.rows, image.cols, image.type());
		cv::Mat_<cv::Vec3b>::iterator itr = result.begin<cv::Vec3b>();

		for (; it != itend; ++it, ++itr) {

			(*itr)[0] = ((*it)[0] & mask) + div / 2;
			(*itr)[1] = ((*it)[1] & mask) + div / 2;
			(*itr)[2] = ((*it)[2] & mask) + div / 2;
		}

		return result;
	}

};


#endif
